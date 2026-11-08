"""Compare bounded FIFO, LRU and LFU on seeded request traces, without a model.

python research/scripts/bench_cache_eviction.py --out research/results/cache_eviction.json
Hit counts are deterministic. Timings measure store calls, not inference or HTTP.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from veya.consistency import _MemoryStore  # noqa: E402


class LRUStore(_MemoryStore):
    """Candidate: retain recently read entries, without changing their TTL."""
    def _get(self, key, now):
        result = super()._get(key, now)
        if result is not None:
            self._data.move_to_end(key)
        return result


class LFUStore:
    """Research candidate for nonexpiring entries; oldest entry breaks frequency ties."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.values = {}
        self.frequencies = {}

    def get(self, key, now):
        entry = self.values.get(key)
        if entry is None:
            return None
        value, frequency = entry
        bucket = self.frequencies[frequency]
        del bucket[key]
        if not bucket:
            del self.frequencies[frequency]
        self.frequencies.setdefault(frequency + 1, OrderedDict())[key] = None
        self.values[key] = (value, frequency + 1)
        return value, math.inf

    def add(self, items, now):
        held = []
        for key, value, expiry in items:
            assert expiry == math.inf, "The LFU research candidate only models nonexpiring entries"
            prior = self.get(key, now)
            if prior is not None:
                held.append(prior[0])
                continue
            if len(self.values) == self.capacity:
                frequency = min(self.frequencies)
                bucket = self.frequencies[frequency]
                victim, _ = bucket.popitem(last=False)
                del self.values[victim]
                if not bucket:
                    del self.frequencies[frequency]
            self.values[key] = (value, 1)
            self.frequencies.setdefault(1, OrderedDict())[key] = None
            held.append(value)
        return held


def traces(capacity, count, seed):
    rng = random.Random(seed)
    zipf = rng.choices(range(capacity * 20), weights=[1 / (i + 1) ** 1.1 for i in range(capacity * 20)], k=count)
    hot_scan = [rng.randrange(max(1, capacity // 5)) if rng.random() < .9 else capacity * 40 + i
                for i in range(count)]
    shift = [(rng.randrange(max(1, capacity * 4 // 5)) + (capacity * 20 if i >= count // 2 else 0))
             if rng.random() < .8 else capacity * 40 + i for i in range(count)]
    return {name: [value.to_bytes(16, "little") for value in values]
            for name, values in (("zipf", zipf), ("hot_plus_scan", hot_scan),
                                 ("phase_shift", shift), ("all_unique", range(count)))}


def replay(factory, capacity, trace):
    store = factory(capacity)
    hits = 0
    start = time.perf_counter_ns()
    for key in trace:
        entry = store.get(key, 0)
        if entry is None:
            store.add([(key, b"first-decision", math.inf)], 0)
        else:
            assert entry[0] == b"first-decision"
            hits += 1
    elapsed = time.perf_counter_ns() - start
    return {"hits": hits, "misses": len(trace) - hits, "hit_rate": hits / len(trace),
            "ns_per_request": elapsed / len(trace)}


def validate_candidates():
    for factory, expected in ((_MemoryStore, None), (LRUStore, (b"a", math.inf))):
        store = factory(2)
        store.add([(b"a", b"a", math.inf), (b"b", b"b", math.inf)], 0)
        assert store.get(b"a", 0) == (b"a", math.inf)
        store.add([(b"c", b"c", math.inf)], 0)
        assert store.get(b"a", 0) == expected
    store = LRUStore(2)
    store.add([(b"a", b"first", 1)], 0)
    assert store.get(b"a", .5) == (b"first", 1)
    assert store.get(b"a", 1) is None, "LRU access must not renew expired entries"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capacities", default="256,1024")
    parser.add_argument("--requests", type=int, default=60000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--production-lru", action="store_true", help="measure the implemented memory-store LRU")
    args = parser.parse_args()
    capacities = [int(value) for value in args.capacities.split(",")]
    if min(capacities) < 2 or args.requests < max(capacities) * 2 or args.repeats < 1:
        parser.error("capacities must be >=2, requests >= twice the largest capacity, repeats >=1")
    validate_candidates()
    factories = {"fifo": _MemoryStore, "lru": LRUStore, "lfu": LFUStore}
    if args.production_lru:
        factories["lru"] = lambda capacity: _MemoryStore(capacity, eviction="lru")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "parameters": vars(args) | {"out": str(args.out)},
              "method": "Same seeded traces, empty stores, no expiry or renewal. Policy order rotates across repeats. "
                        "Only get/add store operations are timed. LFU is a research implementation without aging. "
                        "Workstation load uncontrolled; no inference-speed claim.",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("veya/consistency.py", "research/scripts/bench_cache_eviction.py")},
              "cases": []}
    for capacity in capacities:
        for name, trace in traces(capacity, args.requests, args.seed).items():
            measured = {policy: [] for policy in factories}
            for repeat in range(args.repeats):
                policies = list(factories)
                policies = policies[repeat % 3:] + policies[:repeat % 3]
                for policy in policies:
                    measured[policy].append(replay(factories[policy], capacity, trace))
            for samples in measured.values():
                assert len({sample["hits"] for sample in samples}) == 1
            result = {policy: {**samples[0], "samples_ns_per_request": [s["ns_per_request"] for s in samples],
                               "median_ns_per_request": statistics.median(s["ns_per_request"] for s in samples)}
                      for policy, samples in measured.items()}
            report["cases"].append({"capacity": capacity, "trace": name,
                                    "trace_sha256": hashlib.sha256(b"".join(trace)).hexdigest(), "policies": result})
            print(capacity, name, {p: round(r["hit_rate"] * 100, 2) for p, r in result.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
