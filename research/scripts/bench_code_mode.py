"""Compare equivalent local MCP operation workflows with and without Code Mode.

Uses an existing local English CPU checkpoint; never downloads weights or calls an LLM.
Reports direct-tool workflow time, actual forwards, outer-call counts and JSON response bytes.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veya import DecisionCache, Router  # noqa: E402
from veya.mcp import tools  # noqa: E402
from veya.mcp.code_mode import CodeModeRunner, code_mode_description  # noqa: E402


QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does this request need immediate attention?"}}
FOLLOWUP = {"review": {"type": "noul", "instructions": "Should a human review this request?"}}
SCRIPT = """async input => {
  const first = await veya.predict_batch({requests: input.requests});
  const result = first.requests.map(r => ({urgent:r.answers.urgent.noul}));
  if (input.dependent) {
    const next = input.requests.map((r,i) => ({...r,
      state:{message:r.state.message+' Stage-one urgency: '+(result[i].urgent>=.5?'high.':'low.')},
      questions:input.followup}));
    const second = await veya.predict_batch({requests:next});
    second.requests.forEach((r,i) => {result[i].review=r.answers.review.noul;});
  }
  return result;
}"""


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def sources():
    paths = sorted((ROOT / "laya").rglob("*.py")) + [Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def catalog_sizes():
    from veya.mcp.server import create_code_server, server
    runner = CodeModeRunner()
    try:
        code_server = create_code_server(runner)
        classic_tools = asyncio.run(server.list_tools())
        code_tools = asyncio.run(code_server.list_tools())
        return {"classic_tools": len(classic_tools), "code_tools": len(code_tools),
                "classic_catalog_json_bytes": encoded_size([tool.model_dump(mode="json") for tool in classic_tools]),
                "code_catalog_json_bytes": encoded_size([tool.model_dump(mode="json") for tool in code_tools])}
    finally:
        runner.close()


def classic(router, requests, dependent, singles):
    response_bytes = 0
    calls = 0

    def score(rows):
        nonlocal calls, response_bytes
        if singles:
            values = [tools.laya_predict(**row, router=router) for row in rows]
            calls += len(values)
            response_bytes += sum(encoded_size(value) for value in values)
        else:
            batch = tools.laya_predict_batch(rows, router=router)
            calls += 1
            response_bytes += encoded_size(batch)
            values = batch["requests"]
        return values

    first = score(requests)
    output = [{"urgent": row["answers"]["urgent"]["noul"]} for row in first]
    if dependent:
        second = score([{**row, "questions": FOLLOWUP, "state": {"message": row["state"]["message"]
                        + " Stage-one urgency: " + ("high." if output[index]["urgent"] >= .5 else "low.")}}
                        for index, row in enumerate(requests)])
        for result, row in zip(output, second):
            result["review"] = row["answers"]["review"]["noul"]
    return output, calls, response_bytes


def run_case(agent, requests, scenario, mode):
    cache = DecisionCache(required=True, maxsize=None, fingerprint="code-benchmark-v1")
    router = Router(hooks=[cache], performance_mode="default")
    router.attach("english", agent)
    runner = CodeModeRunner() if mode == "code" else None
    dependent = scenario == "two_stage"
    if scenario == "prewarmed_replay":
        tools.laya_predict_batch(requests, router=router)
    counters = {"calls": 0, "rows": 0}

    def forward(_model, inputs):
        counters["calls"] += 1
        counters["rows"] += int(inputs[0].shape[0])

    handle = agent.model.register_forward_pre_hook(forward)
    before = cache.cache_info()
    try:
        begun = time.perf_counter_ns()
        if runner is not None:
            output = runner.execute(SCRIPT, {"requests": requests, "dependent": dependent, "followup": FOLLOWUP}, router=router)
            calls, response_bytes = 1, encoded_size(output)
        else:
            output, calls, response_bytes = classic(router, requests, dependent, mode == "classic_singles")
        elapsed = (time.perf_counter_ns() - begun) / 1_000_000
        after = cache.cache_info()
        if scenario == "prewarmed_replay":
            assert counters["calls"] == 0, counters
        return {"elapsed_ms": elapsed, "input_requests_per_second": len(requests) * 1000 / elapsed,
                "outer_operation_calls": calls, "response_json_bytes": response_bytes,
                "forward_calls": counters["calls"], "forward_rows": counters["rows"], "output": output,
                "cache": {key: after[key] - before[key] for key in ("hits", "misses", "conflicts", "errors")}}
    finally:
        handle.remove()
        if runner is not None:
            runner.close()
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    path = args.checkpoint.resolve()
    if not (path / "model.safetensors").is_file() or not 1 <= args.requests <= 64 or args.repeats < 1:
        parser.error("use an existing local model.safetensors checkpoint, 1..64 requests and positive repeats")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
    import torch
    from veya import Agent
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    agent = Agent(str(path), device="cpu", backend="eager", performance_mode="default")
    agent.system_one({"message": "Please review the refund request."}, QUESTIONS)
    original_sources = sources()
    requests = [{"state": {"message": "Ticket %d: Please review the payment refund request today." % index},
                 "questions": QUESTIONS, "model": "english"} for index in range(args.requests)]
    artifacts = {file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                 for file in sorted(path.rglob("*")) if file.is_file()
                 and (file.suffix == ".json" or file.name == "model.safetensors" or file.name == "vocab.txt")}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "torch": torch.__version__, "quickjs_ng": version("quickjs-ng"), "psutil": version("psutil"),
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "backend": "English CPU fp32 eager", "requests": args.requests, "repeats": args.repeats,
              "artifacts": artifacts, "source_sha256": original_sources, "cases": [],
              "code_description_utf8_bytes": len(code_mode_description().encode("utf-8")),
              "catalog": catalog_sizes(),
              "method": "One warmed resident checkpoint. Each sample uses a fresh required in-memory cache. "
                        "Order rotates across modes; CPU thread count is four. Every code sample creates a fresh "
                        "sandbox process, so timing includes its startup, JSON IPC, validation and host work. "
                        "Classic samples call the same validated functions directly. JSON response bytes exclude "
                        "MCP envelopes and are not token counts. No LLM or remote-network round trips are simulated. "
                        "Two-stage uses dependent awaiting and returns a compact result; classic batching already "
                        "needs only two calls. Nearest-rank latency percentiles; at three samples p95/p99 are maxima. "
                        "Input throughput counts initial records, not follow-up questions. Peak memory is not measured. "
                        "Small sample, uncontrolled workstation load; no general speed claim."}
    modes = ["classic_singles", "classic_batch", "code"]
    for scenario in ("unique_batch", "two_stage", "prewarmed_replay"):
        samples = []
        for repeat in range(args.repeats):
            row = {"repeat": repeat}
            for mode in modes[repeat % 3:] + modes[:repeat % 3]:
                row[mode] = run_case(agent, requests, scenario, mode)
            reference = row["classic_batch"]["output"]
            row["code_exact_output_match"] = row["code"]["output"] == reference
            row["single_vs_batch_max_probability_delta"] = max(
                abs(value - reference[index][key]) for index, item in enumerate(row["classic_singles"]["output"])
                for key, value in item.items())
            assert row["code_exact_output_match"], "same batch computation changed between classic and Code Mode"
            samples.append(row)
        summary = {mode: {key: statistics.median(row[mode][key] for row in samples)
                          for key in ("elapsed_ms", "input_requests_per_second", "outer_operation_calls", "response_json_bytes", "forward_calls", "forward_rows")}
                   for mode in modes}
        for mode in modes:
            values = [row[mode]["elapsed_ms"] for row in samples]
            summary[mode].update(p50_ms=percentile(values, .5), p95_ms=percentile(values, .95), p99_ms=percentile(values, .99))
        report["cases"].append({"scenario": scenario, "runs": samples, "summary": summary})
        print(scenario, json.dumps(summary), flush=True)
    assert sources() == original_sources, "runtime source changed; rerun after freeze"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
