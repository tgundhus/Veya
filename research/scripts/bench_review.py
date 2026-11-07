"""Reproducible CPU integration smoke and latency measurement on a local checkpoint.

Run: python research/scripts/bench_review.py --checkpoint PATH --out report.json
This checks consistency and execution paths. It is not an accuracy evaluation.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import transformers  # noqa: E402

import veya  # noqa: E402
from veya import Agent, DecisionCache  # noqa: E402
from veya.agent import _checkpoint_signature  # noqa: E402
from veya.revisions import snapshot_revision  # noqa: E402

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments and refunds", "technical": "bugs and outages",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "a critical deadline or blocking issue"]},
    "refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}
STATES = [
    "We were billed twice for March. Please refund the duplicate charge today.",
    "The export crashes every time. Our whole team is blocked and needs a fix today.",
    "Thanks for your help. Everything works now and there is no further action needed.",
    "Please explain how to update the account settings when you have time. " * 6,
]


def timed(call, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def decisions(result):
    return {name: answer.get("choice", round(answer.get("score", 0))) if answer["type"] != "noul"
            else answer["noul"] >= 0.5 for name, answer in result["answers"].items()}


def probabilities(result):
    return {name: answer.get("probabilities", {"true": answer.get("noul")})
            for name, answer in result["answers"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    agent = Agent(args.checkpoint, device="cpu")
    load_ms = (time.perf_counter() - started) * 1000
    print("Checkpoint loaded in %.1f ms" % load_ms, flush=True)
    reference = [agent.predict(state, QUESTIONS) for state in STATES]
    repeated = [agent.predict(state, QUESTIONS) for state in STATES]
    batched = agent.predict_batch(STATES, QUESTIONS, batch_size=2, sort_by_length=True)
    repeated_identical = all(left == right for left, right in zip(reference, repeated))
    decision_flips = sum(decisions(left)[name] != decisions(right)[name]
                         for left, right in zip(reference, batched) for name in left["answers"])
    max_delta = max(abs(value - probabilities(right)[name][label])
                    for left, right in zip(reference, batched)
                    for name, distribution in probabilities(left).items() for label, value in distribution.items())
    assert repeated_identical, "identical CPU calls changed their payloads"
    assert decision_flips == 0, "batching changed a decision in the smoke workload"
    report = {
        "environment": {"laya": veya.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                        "platform": platform.platform(), "processor": platform.processor(), "threads": args.threads,
                        "device": str(agent.device), "dtype": str(agent.dtype), "amp_enabled": agent.amp_enabled,
                        "checkpoint": "convaiinnovations/laya" if snapshot_revision(args.checkpoint) else "local checkpoint",
                        "revision": snapshot_revision(args.checkpoint)},
        "workload": {"states": len(STATES), "questions_per_state": len(QUESTIONS), "repeats": args.repeats,
                     "accuracy_evaluation": False},
        "timing_note": "Shared workstation load is uncontrolled. Timings are exploratory CPU measurements; "
                       "the small workload verifies integration and consistency, not predictive accuracy.",
        "load_ms": load_ms,
        "local_fingerprint": timed(lambda: _checkpoint_signature(args.checkpoint,
                                    str(Path(args.checkpoint) / "model.safetensors"), {}), 20),
        "single": timed(lambda: agent.predict(STATES[0], QUESTIONS), args.repeats),
        "four_serial": timed(lambda: [agent.predict(state, QUESTIONS) for state in STATES], args.repeats),
        "four_batched": timed(lambda: agent.predict_batch(STATES, QUESTIONS, batch_size=2, sort_by_length=True), args.repeats),
        "consistency": {"repeat_payloads_identical": repeated_identical, "batch_decision_flips": decision_flips,
                        "batch_max_probability_delta": max_delta},
        "answers": [result["answers"] for result in reference],
    }
    print("Single median %.2f ms; four-state batch %.2f ms" %
          (report["single"]["median_ms"], report["four_batched"]["median_ms"]), flush=True)
    long_state = (STATES[0] + " ") * 40
    scan = agent.predict_long(long_state, QUESTIONS, window=80, stride=40, batch_size=2)
    assert scan["usage"]["windows"] > 1
    assert all("window" in answer for answer in scan["answers"].values())
    assert not scan["usage"].get("truncated"), "the scanned windows were truncated again"
    report["long"] = {"windows": scan["usage"]["windows"], "usage": scan["usage"],
                      **timed(lambda: agent.predict_long(long_state, QUESTIONS, window=80, stride=40, batch_size=2),
                              args.repeats)}
    rows = []
    forward = agent._forward

    def counted(batch):
        rows.append(batch["input_ids"].shape[0])
        return forward(batch)

    agent._forward = counted
    with tempfile.TemporaryDirectory() as directory:
        cache_path = str(Path(directory) / "decisions.sqlite")
        cache = DecisionCache(cache_path, ttl=3600)
        agent.hooks = (cache,)
        first = agent.predict(STATES[0], QUESTIONS)
        forward_count = len(rows)
        report["sqlite_replay"] = timed(lambda: agent.predict(STATES[0], QUESTIONS), args.repeats * 5)
        assert agent.predict(STATES[0], QUESTIONS) == first
        assert len(rows) == forward_count, "a replay ran the model"
        cache.close()
        reopened = DecisionCache(cache_path, ttl=3600)
        agent.hooks = (reopened,)
        assert agent.predict(STATES[0], QUESTIONS) == first
        assert len(rows) == forward_count, "reopening the persisted cache ran the model"
        report["sqlite_replay"]["counters_after_reopen"] = reopened.cache_info()
        reopened.close()
    report["sqlite_replay"]["forward_passes_avoided"] = args.repeats * 5 + 2
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", target, flush=True)


if __name__ == "__main__":
    main()
