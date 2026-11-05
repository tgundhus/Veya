"""Compare queue-inclusive HTTP hits/misses with optional verified cache bypass.

python research/scripts/bench_http_replay.py --out research/results/http_replay.json
Add --checkpoint /local/laya --requests 16 for existing English CPU fp32 weights.
No model is downloaded. The synthetic default measures orchestration only.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_microbatch import QUESTIONS, count_agent, fixture_agent, percentile  # noqa: E402
from laya import DecisionCache, Router  # noqa: E402
from laya.checkpoint import _checkpoint_signature  # noqa: E402
from laya.serve import create_app  # noqa: E402


def sources():
    files = sorted((ROOT / "laya").rglob("*.py")) + [Path(__file__).resolve(),
                                                    Path(__file__).with_name("bench_microbatch.py")]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def requests(name, count):
    rows, warm = [], set()
    for i in range(count):
        hot = name == "all_hits" or name == "mixed75" and i % 4 != 3
        state = ("hot%d" % (i % 3) if hot else "fresh%d" % i) + ": Please review the refund request urgently."
        rows.append({"state": state, "questions": QUESTIONS, "model": "english"})
        if hot:
            warm.add(state)
    return rows, warm


async def drive(rows, warm, enabled, concurrency, agent, checkpoint, graph, directory):
    import httpx
    import time

    os.environ["LAYA_HTTP_REPLAY_FAST_PATH"] = "1" if enabled else "0"

    cache = DecisionCache(str(directory / "decisions.sqlite"), required=True, maxsize=None,
                          fingerprint="http-replay-benchmark-v1")
    options = {"models": {"english": str(checkpoint)}, "performance_mode": "default"}
    if graph is not None:
        options.update(backend="onnx", onnx_paths={"english": str(graph)})
    primer = Router(hooks=[cache], **options)
    primer.attach("english", agent)
    for state in sorted(warm):
        primer.predict(state, QUESTIONS, model="english")
    manifest = directory / "policy.json"
    primer.export_replay_manifest(str(manifest))
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)
    router.attach("english", agent)
    for field in ("calls", "rows", "forward_calls", "forward_rows"):
        if hasattr(agent, field):
            setattr(agent, field, 0)
    if hasattr(agent, "forward_shapes"):
        agent.forward_shapes = []
    before = cache.cache_info()
    app = create_app(router)
    limit = asyncio.Semaphore(concurrency)
    latencies, canonical = [], [None] * len(rows)
    request_latencies, cache_headers = [None] * len(rows), [False] * len(rows)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                async def one(index, row):
                    async with limit:
                        begun = time.perf_counter()
                        response = await client.post("/v1/systemone", json=row)
                        elapsed_ms = (time.perf_counter() - begun) * 1000
                        latencies.append(elapsed_ms)
                        request_latencies[index] = elapsed_ms
                    if response.status_code != 200:
                        raise RuntimeError("benchmark request failed: %s" % response.status_code)
                    canonical[index] = response.json()
                    cache_headers[index] = response.headers.get("X-Decision-Cache") == "hit"
                begun = time.perf_counter()
                await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
                elapsed = time.perf_counter() - begun
        after = cache.cache_info()
        stats = {"requests": len(rows), "elapsed_seconds": elapsed,
                 "requests_per_second": len(rows) / elapsed, "latencies_ms": latencies,
                 "p50_ms": percentile(latencies, .50), "p95_ms": percentile(latencies, .95),
                 "p99_ms": percentile(latencies, .99), "agent_calls": agent.calls,
                 "agent_requested_states": agent.rows, "fast_path_hits": sum(cache_headers),
                 "request_latencies_ms": request_latencies, "fast_path_hits_by_request": cache_headers,
                 "cache_request_counters": {key: after[key] - before[key]
                                            for key in ("hits", "misses", "conflicts", "coalesced", "errors")}}
        for label, selected in (("prewarmed", True), ("new", False)):
            samples = [latency for row, latency in zip(rows, request_latencies) if (row["state"] in warm) == selected]
            stats[label + "_latencies_ms"] = samples
            if samples:
                stats.update({label + "_p50_ms": percentile(samples, .50), label + "_p95_ms": percentile(samples, .95)})
        if hasattr(agent, "forward_calls"):
            stats.update(model_forward_calls=agent.forward_calls, model_forward_rows=agent.forward_rows,
                         model_forward_shapes=[list(shape) for shape in agent.forward_shapes])
            assert len(agent.forward_shapes) == agent.forward_calls
            assert sum(shape[0] for shape in agent.forward_shapes) == agent.forward_rows
        return stats, canonical
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--wait-ms", type=float, default=1)
    args = parser.parse_args()
    if min(args.requests, args.repeats, args.concurrency) < 1:
        parser.error("requests, repeats and concurrency must be positive")
    import math
    if not math.isfinite(args.wait_ms) or args.wait_ms < 0:
        parser.error("wait-ms must be finite and nonnegative")
    os.environ.update(LAYA_MICROBATCH_MAX_STATES="8", LAYA_MICROBATCH_WAIT_MS=str(args.wait_ms),
                      LAYA_MAX_CONCURRENT=str(max(16, args.concurrency)), LAYA_MAX_BATCH_TOKENS="131072")
    for name in ("LAYA_API_KEY", "LAYA_TOURNAMENT_GROUP", "LAYA_JEV_STRICT", "LAYA_IDLE_UNLOAD_SECONDS",
                 "LAYA_REPLAY_MANIFEST", "LAYA_SHA256_DIGESTS", "LAYA_REVISION"):
        os.environ.pop(name, None)
    runtime = {"processor": platform.processor(), "logical_cpus": os.cpu_count(),
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")}
    actual, identity, graph = None, {}, None
    if args.checkpoint:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
        checkpoint = args.checkpoint.resolve()
        if not (checkpoint / "model.safetensors").is_file():
            parser.error("checkpoint must contain existing local weights")
        import torch
        from laya import Agent
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        actual = Agent(str(checkpoint), device="cpu", backend="eager", performance_mode="default")
        actual.system_one("Please review the refund request urgently.", QUESTIONS)
        actual = count_agent(actual)
        actual.forward_shapes = []

        def record_shapes(_module, inputs):
            actual.forward_shapes.append([int(size) for size in inputs[0].shape])

        actual.model.register_forward_pre_hook(record_shapes)
        runtime.update(torch=torch.__version__, cpu_threads=torch.get_num_threads(),
                       cpu_interop_threads=torch.get_num_interop_threads(), amp="fp32", backend="eager")
        files = {checkpoint / "model.safetensors", *checkpoint.rglob("*.json")}
        files.update(path for path in (checkpoint / "tokenizer").rglob("*") if path.is_file())
        identity = {str(path.relative_to(checkpoint)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(files)}
    initial_sources = sources()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": platform.python_version(), "runtime": runtime, "checkpoint_sha256": identity,
              "source_sha256": initial_sources, "parameters": {"requests": args.requests,
              "repeats": args.repeats, "concurrency": args.concurrency, "microbatch_max_states": 8,
              "microbatch_wait_ms": args.wait_ms, "required": True, "ttl": None, "maxsize": None},
              "backend": "English CPU fp32" if actual else "controlled fixed-overhead fixture",
              "method": "Local ASGI, warmed resident model, fresh SQLite FULL store per paired run. "
                        "Both modes use the same approved manifest; only the HTTP fast-path flag changes. "
                        "Alternating run order. Client latency includes server queues/JSON; client concurrency "
                        "semaphore wait excluded. Nearest-rank percentiles. Source hashes checked before/after. "
                        "Paired payload equality, probability drift and label flips are recorded. "
                        "Request latencies and cache-hit flags preserve input order; prewarmed/new slices are separate. "
                        "Actual Torch input tensor shapes are recorded for every forward attempt. "
                        "Prewarmed hits must match exactly. No TCP or model-load timing. "
                        "Small tail sample and uncontrolled machine load limit interpretation. "
                        "Synthetic timings measure orchestration, not neural inference.", "scenarios": []}
    with tempfile.TemporaryDirectory(prefix="veya-http-replay-", dir=ROOT / ".venv") as scratch:
        directory = Path(scratch)
        if actual is None:
            try:
                import onnx
            except ImportError:
                parser.error("the controlled ONNX manifest fixture needs the onnx extra; alternatively use --checkpoint")
            checkpoint = directory / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "rl_agent_config.json").write_text("{}", encoding="utf-8")
            graph = checkpoint / "model.onnx"
            onnx.save(onnx.helper.make_model(onnx.helper.make_graph([], "identity-fixture", [], [])), str(graph))
            actual = fixture_agent()
            actual._checkpoint_signature = _checkpoint_signature(str(checkpoint), str(graph), {})
        for scenario in ("all_hits", "mixed75", "all_misses"):
            rows, warm = requests(scenario, args.requests)
            result = {"name": scenario, "prewarmed_request_fraction": sum(row["state"] in warm for row in rows) / len(rows),
                      "prewarmed_by_request": [row["state"] in warm for row in rows],
                      "runs": []}
            for repeat in range(args.repeats):
                paired, outputs = {}, {}
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    run_dir = directory / ("%s-%s-%s" % (scenario, repeat, enabled))
                    run_dir.mkdir()
                    stats, payloads = asyncio.run(drive(rows, warm, enabled, args.concurrency,
                                                       actual, checkpoint, graph, run_dir))
                    paired["fast_path" if enabled else "queued"] = stats
                    outputs[enabled] = payloads
                    if scenario == "all_hits" and stats["agent_requested_states"] != 0:
                        raise RuntimeError("a prewarmed hit computed another state")
                paired["outputs_equal"] = outputs[False] == outputs[True]
                paired["max_probability_delta"] = max(abs(a["answers"]["q"]["noul"] - b["answers"]["q"]["noul"])
                                                        for a, b in zip(outputs[False], outputs[True]))
                paired["label_flips"] = sum((a["answers"]["q"]["noul"] >= .5) != (b["answers"]["q"]["noul"] >= .5)
                                             for a, b in zip(outputs[False], outputs[True]))
                if any(a != b for row, a, b in zip(rows, outputs[False], outputs[True]) if row["state"] in warm):
                    raise RuntimeError("verified replay changed a prewarmed response payload")
                result["runs"].append(paired)
            metrics = ["requests_per_second", "p50_ms", "p95_ms", "p99_ms", "agent_calls", "agent_requested_states"]
            if hasattr(actual, "forward_calls"):
                metrics += ["model_forward_calls", "model_forward_rows"]
            for label in ("prewarmed", "new"):
                metrics += [label + suffix for suffix in ("_p50_ms", "_p95_ms")
                            if label + suffix in result["runs"][0]["queued"]]
            result["summary"] = {mode: {metric: statistics.median(run[mode][metric] for run in result["runs"])
                                        for metric in metrics} for mode in ("queued", "fast_path")}
            result["all_outputs_equal"] = all(run["outputs_equal"] for run in result["runs"])
            report["scenarios"].append(result)
            print(scenario, json.dumps(result["summary"]), flush=True)
    if sources() != initial_sources:
        raise RuntimeError("runtime source changed during benchmark; rerun on frozen source")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
