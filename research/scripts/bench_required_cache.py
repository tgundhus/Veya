"""Measure optional/required decision-cache orchestration without a neural model.

Run: python research/scripts/bench_required_cache.py --out research/results/required_cache.json
Reports exact-hit and unique-commit latency on memory and local SQLite stores.
These are synchronous Router/cache timings, not model inference or HTTP throughput.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veya import DecisionCache, Router  # noqa: E402


QUESTIONS = {
    "decision": {"type": "choice", "instructions": "Which outcome applies to this request?",
                 "criteria": {"allow": "allow the request", "review": "review the request"}},
}
SOURCE_FILES = ("veya/consistency.py", "veya/hooks.py", "veya/router.py", "veya/errors.py",
                "veya/inference.py", "veya/confidence.py", "veya/lang.py", "veya/performance.py",
                "research/scripts/bench_required_cache.py")


class LocalPredictor:
    """A caller-owned deterministic stand-in; no weights, tokenizer or inference runtime."""

    model_id = "veya-cache-benchmark-fixture"
    revision = "fixture-v1"
    cfg = {"max_len": 512, "head_max_len": 192}
    hooks = ()

    def __init__(self):
        self.probability = 0.51
        self.forward_calls = 0

    def system_one(self, state, questions, **kwargs):
        self.forward_calls += 1
        probability = self.probability
        return {"model": self.model_id, "answers": {
            "decision": {"type": "choice", "choice": "allow" if probability >= 0.5 else "review",
                         "probabilities": {"allow": probability, "review": 1 - probability},
                         "answer_confidence": max(probability, 1 - probability)}},
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def distribution(samples):
    ordered = sorted(samples)
    result = {"count": len(ordered)}
    for label, fraction in (("p50_us", 0.5), ("p95_us", 0.95), ("p99_us", 0.99)):
        result[label] = round(ordered[math.ceil(len(ordered) * fraction) - 1] / 1000, 3)
    result["mean_us"] = round(statistics.mean(ordered) / 1000, 3)
    return result


def run_repeat(storage, required, repeat, args, directory):
    path = str(directory / ("%s-%d.sqlite" % (storage, repeat))) if storage == "sqlite" else None
    cache = DecisionCache(path, ttl=None, maxsize=None, fingerprint="benchmark-policy-v1", required=required)
    router = Router(hooks=[cache], performance_mode="default")
    predictor = LocalPredictor()
    router.attach("english", predictor)

    def predict(state):
        return router.predict(state, QUESTIONS, model="english")

    try:
        synchronous = None if path is None else cache._store._db.execute("PRAGMA synchronous").fetchone()[0]
        journal = None if path is None else cache._store._db.execute("PRAGMA journal_mode").fetchone()[0]
        if path is not None:
            assert synchronous == (2 if required else 1), (storage, required, synchronous)
            assert journal.lower() == "wal", journal
        first = predict("Stable exact request")
        assert first["answers"]["decision"]["probabilities"]["allow"] == 0.51
        predictor.probability = 0.49
        for _ in range(args.warmups):
            assert predict("Stable exact request")["answers"] == first["answers"]
        phases = {}
        for phase, count in (("warm_exact_hits", args.hits), ("unique_commits", args.commits)):
            before_info = cache.cache_info()
            before_calls = predictor.forward_calls
            samples = []
            returned_errors = 0
            expected = first["answers"] if phase == "warm_exact_hits" else None
            for index in range(count):
                state = "Stable exact request" if expected is not None else "Unique request %06d" % index
                started = time.perf_counter_ns()
                try:
                    result = predict(state)
                except Exception:
                    returned_errors += 1
                    raise
                finally:
                    samples.append(time.perf_counter_ns() - started)
                if expected is not None:
                    assert result["answers"] == expected, (phase, storage, required, index)
                else:
                    assert result["answers"]["decision"]["choice"] == "review", (storage, required, index)
            after_info = cache.cache_info()
            forward_calls = predictor.forward_calls - before_calls
            assert forward_calls == (0 if phase == "warm_exact_hits" else count), forward_calls
            assert after_info["errors"] == before_info["errors"], after_info
            phases[phase] = {**distribution(samples), "samples_ns": samples, "forward_calls": forward_calls,
                             "returned_errors": returned_errors,
                             "cache_errors": after_info["errors"] - before_info["errors"],
                             "hits": after_info["hits"] - before_info["hits"],
                             "misses": after_info["misses"] - before_info["misses"]}
        return {"repeat": repeat, "sqlite_synchronous": synchronous, "sqlite_journal_mode": journal,
                "first_writer_replay": {"first_probability": 0.51, "replacement_probability": 0.49,
                                        "replayed_probability": 0.51, "all_hits_match": True},
                **phases}
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".venv")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--hits", type=int, default=500)
    parser.add_argument("--commits", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=20)
    args = parser.parse_args()
    if min(args.repeats, args.hits, args.commits) < 1 or args.warmups < 1:
        parser.error("repeats, hits, commits and warmups must be positive")
    work_dir = args.work_dir.resolve()
    if not work_dir.is_dir():
        parser.error("work-dir must be an existing directory")
    initial_sources = source_hashes()
    results = {(storage, required): [] for storage in ("memory", "sqlite") for required in (False, True)}
    for storage in ("memory", "sqlite"):
        for repeat in range(args.repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for required in order:
                with tempfile.TemporaryDirectory(prefix="veya-cache-bench-", dir=work_dir) as tmp:
                    directory = Path(tmp).resolve()
                    assert directory.is_relative_to(work_dir)
                    row = run_repeat(storage, required, repeat, args, directory)
                results[(storage, required)].append(row)
    assert source_hashes() == initial_sources, "benchmark source changed during execution; rerun after edits finish"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": initial_sources, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "sqlite": sqlite3.sqlite_version,
        "parameters": {"repeats": args.repeats, "hits_per_repeat": args.hits,
                       "commits_per_repeat": args.commits, "warmups_per_repeat": args.warmups,
                       "ttl": None, "maxsize": None, "fingerprint": "benchmark-policy-v1",
                       "coalesce": True, "question_schema": QUESTIONS},
        "method": "One synchronous Router request at a time. Each repeat creates a fresh store. "
                  "Optional/required order alternates each repeat. Individual call latency uses perf_counter_ns; "
                  "percentiles use nearest rank. Result assertions occur outside timed sections. "
                  "SQLite uses WAL; optional uses NORMAL (1), required uses FULL (2). "
                  "Raw timings include Python routing/hooks/key/serialization/store work and the fake predictor. "
                  "No neural model, tokenizer, HTTP, MCP or GPU; no inference-speed claim. "
                  "Machine load and filesystem caches are uncontrolled. Temporary databases are removed after each run.",
        "cases": [], "comparisons": [],
    }
    for (storage, required), rows in results.items():
        case = {"storage": storage, "required": required, "repeats": rows}
        for phase in ("warm_exact_hits", "unique_commits"):
            samples = [sample for row in rows for sample in row[phase]["samples_ns"]]
            case[phase] = {**distribution(samples), **{
                name: sum(row[phase][name] for row in rows)
                for name in ("forward_calls", "returned_errors", "cache_errors", "hits", "misses")}}
        report["cases"].append(case)
        print(storage, "required=%s" % required, "hit", case["warm_exact_hits"],
              "commit", case["unique_commits"], flush=True)
    for storage in ("memory", "sqlite"):
        optional, required = [next(row for row in report["cases"]
                                   if row["storage"] == storage and row["required"] == flag)
                              for flag in (False, True)]
        report["comparisons"].append({"storage": storage, **{
            phase + "_required_over_optional_p50": round(required[phase]["p50_us"] / optional[phase]["p50_us"], 3)
            for phase in ("warm_exact_hits", "unique_commits")}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
