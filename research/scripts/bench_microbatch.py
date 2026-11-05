"""Queue-inclusive local HTTP microbatch comparison; synthetic fixture by default.

Use --checkpoint /local/laya for an existing English checkpoint. No models are downloaded.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from laya import DecisionCache, Router  # noqa: E402
from laya.serve import create_app  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402


QUESTIONS = {"q": {"type": "noul", "instructions": "Does this request need immediate attention?"}}


class FixtureAgent:
    cfg = {"max_len": 512, "head_max_len": 192}
    model_id = "controlled-microbatch-fixture"
    revision = "fixture-v1"
    hooks = ()

    def __init__(self):
        self.calls = 0
        self.rows = 0

    def result(self, state, questions):
        return {"model": "fixture", "answers": {
            key: {"type": "noul", "noul": 0.51, "answer_confidence": 0.51} for key in questions},
            "usage": {"input_tokens": len(state) // 4, "output_tokens": 0}}

    def system_one(self, state, questions, **kwargs):
        self.calls += 1
        self.rows += 1
        time.sleep(0.003 + 0.0001 * len(questions))
        return self.result(state, questions)

    def predict_batch(self, states, questions, batch_size=None, **kwargs):
        outputs = []
        size = batch_size or len(states)
        for start in range(0, len(states), size):
            chunk = states[start:start + size]
            self.calls += 1
            self.rows += len(chunk)
            time.sleep(0.003 + 0.0001 * len(chunk) * len(questions))
            outputs.extend(self.result(state, questions) for state in chunk)
        return outputs


def fixture_agent():
    """Use a built-in type with controlled methods, rather than broaden server eligibility."""
    agent = object.__new__(ONNXAgent)
    agent.__dict__.update(vars(FixtureAgent()))
    for name in ("cfg", "model_id", "revision"):
        setattr(agent, name, getattr(FixtureAgent, name))
    for name in ("result", "system_one", "predict_batch"):
        setattr(agent, name, MethodType(getattr(FixtureAgent, name), agent))
    return agent


def count_agent(agent):
    """Count the underlying batch API plus actual Torch forwards, retaining the built-in type."""
    batch = agent.predict_batch

    def predict_batch(states, questions, **kwargs):
        agent.calls += 1
        agent.rows += len(states)
        return batch(states, questions, **kwargs)

    def forward(_module, inputs):
        agent.forward_calls += 1
        agent.forward_rows += int(inputs[0].shape[0])

    agent.calls = agent.rows = agent.forward_calls = agent.forward_rows = 0
    agent.predict_batch = predict_batch
    agent.model.register_forward_pre_hook(forward)
    return agent


def percentile(values, p):
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(len(values) * p) - 1))]


def trace(name, n):
    rows = []
    for i in range(n):
        hot = name == "repeats70" and i % 10 < 7
        identity = "identical" if name == "identical_burst" else ("hot%d" % (i % 4) if hot else "unique%d" % i)
        questions = QUESTIONS
        model = "english"
        if name == "mixed_groups":
            questions = {"q%d" % j: {"type": "noul", "instructions": "Is this urgent?"}
                         for j in range(1 + (i // 4) % 3)}
            model = "english" if (i // 8) % 2 == 0 else "multilingual"
        state = identity + ": " + "Please review the payment refund request. " * (1 + i % 3)
        # Repeats must actually have identical state bytes, not merely similar text.
        if hot or name == "identical_burst":
            state = identity + ": Please review the payment refund request."
        rows.append({"state": state, "questions": questions, "model": model})
    return rows


async def drive(payloads, enabled, concurrency, cache_enabled, actual_agent=None):
    import httpx

    os.environ.update(LAYA_MICROBATCH_MAX_STATES="8" if enabled else "0", LAYA_MICROBATCH_WAIT_MS="1",
                      LAYA_MAX_CONCURRENT=str(max(16, concurrency)), LAYA_MAX_BATCH_TOKENS="131072")
    for name in ("LAYA_API_KEY", "LAYA_TOURNAMENT_GROUP", "LAYA_JEV_STRICT", "LAYA_IDLE_UNLOAD_SECONDS"):
        os.environ.pop(name, None)
    cache = DecisionCache(maxsize=None, fingerprint="microbatch-benchmark-v1") if cache_enabled else None
    router = Router(hooks=[cache] if cache else [], performance_mode="default")
    agents = [actual_agent] if actual_agent else [fixture_agent(), fixture_agent()]
    router.attach("english", agents[0])
    if not actual_agent:
        router.attach("multilingual", agents[1])
    if cache_enabled:
        for i in range(4):
            router.predict("hot%d: Please review the payment refund request." % i, QUESTIONS, model="english")
    cache_before = cache.cache_info() if cache else None
    for agent in agents:
        agent.calls = agent.rows = 0
        if actual_agent:
            agent.forward_calls = agent.forward_rows = 0
    app = create_app(router)
    limit = asyncio.Semaphore(concurrency)
    latencies, canonical = [], [None] * len(payloads)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            async def one(index, payload):
                async with limit:
                    start = time.perf_counter()
                    response = await client.post("/v1/systemone", json=payload)
                    latencies.append((time.perf_counter() - start) * 1000.0)
                if response.status_code != 200:
                    raise RuntimeError("benchmark request failed: %s" % response.status_code)
                canonical[index] = json.dumps(response.json(), sort_keys=True, ensure_ascii=False)
            start = time.perf_counter()
            await asyncio.gather(*(one(i, payload) for i, payload in enumerate(payloads)))
            elapsed = time.perf_counter() - start
    stats = {"requests": len(payloads), "elapsed_seconds": elapsed,
             "requests_per_second": len(payloads) / elapsed,
             "p50_ms": percentile(latencies, .50), "p95_ms": percentile(latencies, .95),
             "p99_ms": percentile(latencies, .99), "latencies_ms": latencies,
             "agent_calls": sum(a.calls for a in agents), "agent_requested_states": sum(a.rows for a in agents)}
    if actual_agent:
        stats.update(model_forward_calls=actual_agent.forward_calls, model_forward_rows=actual_agent.forward_rows)
    if cache:
        stats["cache"] = cache.cache_info()
        stats["cache_preload"] = cache_before
        stats["cache_request_counters"] = {key: stats["cache"][key] - cache_before[key]
                                            for key in ("hits", "misses", "conflicts", "coalesced", "errors")}
        cache.close()
    return stats, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.requests, args.repeats, args.concurrency) < 1:
        parser.error("requests, repeats and concurrency must be positive")
    actual = None
    artifacts = {}
    runtime = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if args.checkpoint:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
        path = args.checkpoint.resolve()
        if not (path / "model.safetensors").is_file():
            parser.error("checkpoint must be a local weight directory")
        import torch
        from laya import Agent
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        runtime.update(torch=torch.__version__, cpu_threads=torch.get_num_threads(),
                       cpu_interop_threads=torch.get_num_interop_threads(), amp="fp32", backend="eager")
        actual = Agent(str(path), device="cpu", backend="eager", performance_mode="default")
        actual.system_one("Please review the payment refund request.", QUESTIONS)
        actual = count_agent(actual)
        identity_files = {path / "model.safetensors", *path.rglob("*.json")}
        identity_files.update(file for file in (path / "tokenizer").rglob("*") if file.is_file())
        artifacts = {str(file.relative_to(path)).replace("\\", "/"): hashlib.sha256(file.read_bytes()).hexdigest()
                     for file in sorted(identity_files)}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": platform.python_version(), "repeats": args.repeats,
              "concurrency": args.concurrency, "backend": "English CPU fp32" if actual else "synthetic fixed-cost agent",
              "artifacts": artifacts, "runtime": runtime,
              "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((ROOT / "laya").glob("*.py")) + [Path(__file__).resolve()]},
              "limits": "Local ASGI transport, no TCP; warmed resident models; wall time includes server queue, "
                        "JSON and routing. Client concurrency wait is excluded from per-request latency. "
                        "Small tail sample; uncontrolled workstation load. Synthetic timing is not model throughput. "
                        "Nearest-rank percentiles. Real Agent calls count the predict_batch boundary (system_one delegates there); "
                        "agent_requested_states counts states entering that API, not hardware rows. Real-checkpoint "
                        "forward calls/rows count the Torch model forward pre-hook, including retry attempts. "
                        "Synthetic calls count the fixture's controlled cost chunks.",
              "microbatch_max_states": 8, "microbatch_wait_ms": 1, "scenarios": []}
    names = ["unique", "repeats70", "identical_burst"] + ([] if actual else ["mixed_groups"])
    for name in names:
        payloads = trace(name, args.requests)
        row = {"name": name, "runs": []}
        if name == "repeats70":
            row["prewarmed_repeat_fraction"] = sum(i % 10 < 7 for i in range(args.requests)) / args.requests
        for repetition in range(args.repeats):
            pair = {}
            outputs = {}
            # Alternate execution order to reduce a systematic warm/load-order advantage.
            for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
                stats, canonical = asyncio.run(drive(payloads, enabled, args.concurrency,
                                                     name in ("repeats70", "identical_burst"), actual))
                if name == "unique" and stats["agent_requested_states"] != len(payloads):
                    raise RuntimeError("uncached unique trace must count each requested state exactly once")
                pair["microbatch" if enabled else "serial"] = stats
                outputs[enabled] = canonical
            pair["outputs_equal"] = outputs[False] == outputs[True]
            row["runs"].append(pair)
        metrics = ["requests_per_second", "p50_ms", "p95_ms", "p99_ms", "agent_calls", "agent_requested_states"]
        if actual:
            metrics.extend(["model_forward_calls", "model_forward_rows"])
        row["summary"] = {mode: {metric: statistics.median(run[mode][metric] for run in row["runs"])
                          for metric in metrics}
                          for mode in ("serial", "microbatch")}
        row["all_outputs_equal"] = all(run["outputs_equal"] for run in row["runs"])
        report["scenarios"].append(row)
        print(name, json.dumps(row["summary"]), "outputs_equal", row["all_outputs_equal"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
