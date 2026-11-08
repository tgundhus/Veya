"""Compare saturated memory-store insertion and mixed expiry against a Git baseline.

Run: python research/scripts/bench_cache_capacity.py --out research/results/cache-capacity.json
This isolates the byte store. It does not measure model inference or end-to-end request latency.
"""
import argparse
import ast
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import veya.consistency as consistency  # noqa: E402


def baseline_store(ref):
    source = subprocess.check_output(["git", "show", ref + ":laya/consistency.py"], cwd=ROOT,
                                     text=True, encoding="utf-8")
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == "_MemoryStore")
    namespace = dict(vars(consistency))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<baseline-store>", "exec"), namespace)
    return namespace["_MemoryStore"]


def saturated(cls, capacity, count):
    store = cls(capacity)
    store.add([(index.to_bytes(16, "little"), b"value", math.inf) for index in range(capacity)], 0)
    started = time.perf_counter()
    for index in range(capacity, capacity + count):
        store.add([(index.to_bytes(16, "little"), b"value", math.inf)], 1)
    return (time.perf_counter() - started) * 1e6 / count


def mixed_expiry(cls, capacity, expired_count=1000, count=5000):
    store = cls(capacity)
    store.add([(index.to_bytes(16, "little"), b"value", 1 if index >= capacity - expired_count else 100)
               for index in range(capacity)], 0)
    started = time.perf_counter()
    for index in range(capacity, capacity + count):
        store.add([(index.to_bytes(16, "little"), b"value", 100)], 2)
    micros = (time.perf_counter() - started) * 1e6 / count
    retained = sum(store.get(index.to_bytes(16, "little"), 2) is not None
                   for index in range(capacity - expired_count))
    return micros, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default="e29aa6d")
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--inserts", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.capacity <= 1000 or args.inserts <= 0 or args.repeats <= 0:
        parser.error("capacity must exceed 1000; inserts and repeats must be positive")
    implementations = {"before": baseline_store(args.baseline), "after": consistency._MemoryStore}
    samples = {name: {"saturated_insert_us": [], "mixed_expiry_insert_us": []} for name in implementations}
    for repetition in range(args.repeats):
        order = list(implementations) if repetition % 2 == 0 else list(reversed(implementations))
        for name in order:
            cls = implementations[name]
            samples[name]["saturated_insert_us"].append(saturated(cls, args.capacity, args.inserts))
            micros, retained = mixed_expiry(cls, args.capacity)
            samples[name]["mixed_expiry_insert_us"].append(micros)
            samples[name]["mixed_expiry_live_entries_retained"] = retained
    for values in samples.values():
        values["saturated_insert_us_median"] = statistics.median(values["saturated_insert_us"])
        values["mixed_expiry_insert_us_median"] = statistics.median(values["mixed_expiry_insert_us"])
    report = {"environment": {"platform": platform.platform(), "processor": platform.processor(),
                              "python": platform.python_version()},
              "baseline_ref": args.baseline, "capacity": args.capacity, "unique_inserts": args.inserts,
              "repeats": args.repeats, "mixed_expiry": {"expired_entries": 1000, "unique_inserts": 5000},
              "timing_note": "Alternating implementations; shared workstation load is uncontrolled. "
                             "Microsecond timings are exploratory, not universal performance claims.",
              **samples}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
