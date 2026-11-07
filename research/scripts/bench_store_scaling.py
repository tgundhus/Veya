"""DecisionCache stores under load: tail latency as they grow, bytes held, threads and processes.

No model is needed; requests go through the cache's hooks over an instant agent (bench_retention.py's
Driver), or straight to a store. Sections:

    tail        latency of every single store call (a hit, and a store on a miss) at 10k, 100k and
                1M held decisions, median to max: the maintenance a store does on the request path
                (the memory store's sweep for expired decisions, SQLite's periodic COUNT(*) and prune)
                shows up in the tail, not the median.
    maintenance one such pass timed on its own at each size, since `tail` stores too few to reach
                the memory store's sweep once it holds more than about 80k decisions.
    renewal     a hit on SQLite when renew_on_hit writes, against one that does not.
    bytes       bytes per held decision for real payloads of 1 to 10 answers, in memory and on disk.
    threads     hits per second through the hooks from 1-16 threads of one process.
    processes   several processes on one SQLite file: throughput, and whether every process got the
                same decision for every request when all of them computed different ones at once.

    python research/scripts/bench_store_scaling.py --out stores.json
    python research/scripts/bench_store_scaling.py --sections tail --sizes 10000,100000,1000000,5000000
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import veya  # noqa: E402
from veya import DecisionCache  # noqa: E402
from veya import consistency  # noqa: E402
from bench_retention import DAY, Driver, QUESTION_SETS, base_payload  # noqa: E402

INF = math.inf


def keys_for(n, salt=b""):
    return [hashlib.blake2b(salt + i.to_bytes(8, "little"), digest_size=16).digest() for i in range(n)]


def dist_us(ns):
    xs = sorted(ns)
    n = len(xs)
    q = lambda p: round(xs[min(n - 1, int(p * n))] / 1e3, 2)  # noqa: E731
    return {"n": n, "median_us": q(0.5), "p99_us": q(0.99), "p99.9_us": q(0.999), "max_us": round(xs[-1] / 1e3, 2),
            "mean_us": round(statistics.mean(xs) / 1e3, 2)}


def payloads():
    """Real result payloads from the archived Feishu run, and larger ones built from them."""
    rows = [json.loads(line) for line in open(os.path.join(ROOT, "research", "benchmarks", "feishu_zh", "results",
                                                             "v1", "laya", "raw.jsonl"), encoding="utf-8")]
    choice = next(r["response"] for r in rows if r["mode"] == "choice")
    four = next(r["response"] for r in rows if r["mode"] == "four_noul")
    answers = list(four["answers"].values()) + list(choice["answers"].values())
    ten = dict(four, answers={"q%d" % i: answers[i % len(answers)] for i in range(10)})
    one = dict(four, answers={"q0": answers[0]})
    return {"1 answer": one, "choice, 4 options": choice, "4 noul answers": four, "10 answers": ten}


# ------------------------------------------------------------------------------------------ tail

def fill(store, keys, blob, now, rng, ttl_days=(1, 30)):
    for start in range(0, len(keys), 2000):
        store.add([(k, blob, now + rng.uniform(*ttl_days) * DAY) for k in keys[start:start + 2000]], now)


def tail(tmp, blob, sizes, ops):
    rng = random.Random(0)
    now = 1_750_000_000.0
    out = {}
    for size in sizes:
        keys = keys_for(size)
        fresh = keys_for(ops, b"fresh")
        path = os.path.join(tmp, "t%d_%%s.sqlite" % size)
        setups = [("memory, maxsize=None", lambda: consistency._MemoryStore(None)),
                  ("memory, maxsize=size", lambda: consistency._MemoryStore(size)),
                  ("sqlite, maxsize=None", lambda: consistency._SQLiteStore(path % "n", None)),
                  ("sqlite, maxsize=size", lambda: consistency._SQLiteStore(path % "b", size))]
        for name, make in setups:
            if size > 1_000_000 and name.startswith("memory"):
                continue
            store = make()
            t = time.perf_counter()
            fill(store, keys, blob, now, rng)
            fill_s = time.perf_counter() - t
            gets, adds = [], []
            probe = [keys[rng.randrange(size)] for _ in range(ops)]
            for i in range(ops):
                t0 = time.perf_counter_ns()
                store.get(probe[i], now)
                t1 = time.perf_counter_ns()
                store.add([(fresh[i], blob, now + rng.uniform(1, 30) * DAY)], now)
                t2 = time.perf_counter_ns()
                gets.append(t1 - t0)
                adds.append(t2 - t1)
            entry = {"held": size, "fill_s": round(fill_s, 1), "get (hit)": dist_us(gets),
                     "add (one miss stored)": dist_us(adds), "held_after": len(store)}
            if name.startswith("sqlite"):
                store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                entry["file_bytes_per_decision"] = round(os.path.getsize(store._db.execute(
                    "PRAGMA database_list").fetchone()[2]) / len(store), 1)
            store.close()
            out.setdefault(str(size), {})[name] = entry
            a = entry["add (one miss stored)"]
            print("  %9d %-22s get p50 %7.2f max %9.1f | add p50 %7.2f p99.9 %9.1f max %10.1f us"
                  % (size, name, entry["get (hit)"]["median_us"], entry["get (hit)"]["max_us"], a["median_us"],
                     a["p99.9_us"], a["max_us"]), flush=True)
    return out


# ----------------------------------------------------------------------------------- maintenance

def maintenance(tmp, blob, sizes):
    """One maintenance pass of each store, timed on its own at each size.

    Both run inside a store call on the request path, holding the store's lock: the memory store
    walks every held decision for expired ones every max(4096, held // 4) stores, and a bounded
    SQLite store counts its rows (and drops the oldest over the bound) every min(1000, maxsize // 10)
    stores. `tail` only meets them when it stores enough to reach one.
    """
    rng = random.Random(1)
    now = 1_750_000_000.0
    out = {}
    for size in sizes:
        keys = keys_for(size, b"maintenance")
        mem = consistency._MemoryStore(None)
        fill(mem, keys, blob, now, rng)
        t = time.perf_counter()
        dropped = mem._sweep(now + 3 * DAY)          # about a tenth of them expired by then
        sweep_ms = (time.perf_counter() - t) * 1e3
        mem.close()
        del mem
        path = os.path.join(tmp, "maint_%d.sqlite" % size)
        sq = consistency._SQLiteStore(path, None)
        fill(sq, keys, blob, now, rng)
        sq.maxsize = size - 1000                      # as if the bound were just passed
        t = time.perf_counter()
        with sq._lock:
            (count,) = sq._db.execute("SELECT COUNT(*) FROM laya_decisions").fetchone()
        count_ms = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        pruned = sq.prune(now)                        # expired none; drops the 1,000 oldest
        prune_ms = (time.perf_counter() - t) * 1e3
        sq.close()
        sweep_every, prune_every = max(4096, size // 4), max(1, min(1000, size // 10))
        out[str(size)] = {"memory sweep ms": round(sweep_ms, 2), "memory sweep dropped": dropped,
                          "memory sweeps every n stores": sweep_every, "sqlite COUNT(*) ms": round(count_ms, 2),
                          "sqlite prune of 1,000 over the bound ms": round(prune_ms, 2), "sqlite pruned": pruned,
                          "sqlite prunes every n stores when bounded": prune_every}
        print("  %9d  memory sweep %8.2f ms (every %d stores) | sqlite COUNT(*) %8.2f ms, prune %8.2f ms (every %d)"
              % (size, sweep_ms, sweep_every, count_ms, prune_ms, prune_every), flush=True)
    return out


# --------------------------------------------------------------------------------------- renewal

def renewal(tmp, payload, n=5000):
    """A hit that writes its renewal against one that does not, through the hooks, on SQLite."""
    out = {}
    clock_t = [1_750_000_000.0]
    consistency._now = lambda: clock_t[0]
    try:
        for label, renew in (("renew_on_hit=False", False), ("renew_on_hit=True, every hit renews", True)):
            cache = DecisionCache(os.path.join(tmp, "renew_%s.sqlite" % renew), ttl=DAY, renew_on_hit=renew)
            drv = Driver(cache, payload)
            q = QUESTION_SETS["billing"]
            for i in range(n):
                drv.request("r%d" % i, q)
            lat = []
            for rnd in range(3):
                clock_t[0] += DAY / 8          # more than ttl/16 later: every hit renews when enabled
                for i in range(n):
                    t = time.perf_counter_ns()
                    drv.request("r%d" % i, q)
                    lat.append(time.perf_counter_ns() - t)
            out[label] = dist_us(lat)
            cache.close()
    finally:
        consistency._now = time.time
    return out


# ----------------------------------------------------------------------------------------- bytes

def bytes_held(tmp, samples, n=20_000):
    out = {}
    enc = DecisionCache()
    for label, payload in samples.items():
        blob = enc._encode(payload)
        keys = keys_for(n, label.encode())
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        mem = consistency._MemoryStore(None)
        for k in keys:
            mem.add([(k, blob, INF)], 1.0)
        heap = tracemalloc.get_traced_memory()[0] - before
        tracemalloc.stop()
        path = os.path.join(tmp, "bytes_%d.sqlite" % len(out))
        store = consistency._SQLiteStore(path, None)
        for start in range(0, n, 2000):
            store.add([(k, blob, INF) for k in keys[start:start + 2000]], 1.0)
        store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        out[label] = {"raw_json_bytes": len(raw), "stored_value_bytes": len(blob),
                      "memory_heap_bytes_per_decision": round(heap / n, 1),
                      "sqlite_file_bytes_per_decision": round(os.path.getsize(path) / n, 1),
                      "per_100k_decisions_memory_mb": round(heap / n * 1e5 / 2 ** 20, 1),
                      "per_1m_decisions_sqlite_mb": round(os.path.getsize(path) / n * 1e6 / 2 ** 20, 1)}
        store.close()
        del mem
    return out


# --------------------------------------------------------------------------------------- threads

def threads(tmp, payload, counts=(1, 2, 4, 8, 16), seconds=2.0, keys=2000):
    out = {}
    for label, path in (("memory", None), ("sqlite", os.path.join(tmp, "threads.sqlite"))):
        cache = DecisionCache(path)
        warm = Driver(cache, payload)
        q = QUESTION_SETS["billing"]
        for i in range(keys):
            warm.request("t%d" % i, q)
        for n in counts:
            done = [0] * n
            stop = threading.Event()

            def work(slot):
                drv = Driver(cache, payload)
                rng = random.Random(slot)
                while not stop.is_set():
                    drv.request("t%d" % rng.randrange(keys), q)
                    done[slot] += 1
            pool = [threading.Thread(target=work, args=(s,)) for s in range(n)]
            for th in pool:
                th.start()
            time.sleep(seconds)
            stop.set()
            for th in pool:
                th.join()
            out.setdefault(label, {})[str(n)] = round(sum(done) / seconds)
        cache.close()
        print("  hits/s by threads, %s: %s" % (label, out[label]), flush=True)
    return out


# ------------------------------------------------------------------------------------- processes

PROC = r"""
import json, os, sys, time, random
sys.path.insert(0, %(root)r); sys.path.insert(0, %(here)r)
from bench_retention import Driver, QUESTION_SETS, base_payload
from veya import DecisionCache
job = json.loads(sys.argv[1])
cache = DecisionCache(job["path"], coalesce=True)
payload = dict(base_payload(), writer=os.getpid())
drv = Driver(cache, payload)
q = QUESTION_SETS["billing"]
while time.time() < job["start_at"]:
    time.sleep(0.0005)
got, lat = {}, []
rng = random.Random(os.getpid())
order = list(range(job["keys"]))
if job["mode"] == "race":
    rng.shuffle(order)
    for i in order:
        result, hit = drv.request("race %%d" %% i, q)
        got[i] = result.get("writer")
else:
    t_end = time.time() + job["seconds"]
    while time.time() < t_end:
        i = rng.randrange(job["keys"]) if rng.random() < job["hit_share"] else rng.randrange(10**9)
        t = time.perf_counter_ns()
        drv.request("mix %%d" %% i, q)
        lat.append(time.perf_counter_ns() - t)
info = cache.cache_info()
print(json.dumps({"got": got, "lat": lat, "conflicts": info["conflicts"], "computed": drv.computed}))
"""


def processes(tmp, counts=(1, 2, 4, 8), race_keys=2000, seconds=3.0, keys=2000):
    script = PROC % {"root": ROOT, "here": HERE}
    out = {"race": {}, "mixed_90pct_hits": {}}

    def launch(n, **job):
        start_at = time.time() + 1.5
        procs = [subprocess.Popen([sys.executable, "-c", script, json.dumps(dict(job, start_at=start_at))],
                                  stdout=subprocess.PIPE, text=True) for _ in range(n)]
        return [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]

    for n in counts:
        # Every process computes its own decision (tagged with its pid) for the same requests at once.
        path = os.path.join(tmp, "race_%d.sqlite" % n)
        res = launch(n, path=path, mode="race", keys=race_keys)
        disagree = sum(1 for i in map(str, range(race_keys)) if len({r["got"][i] for r in res}) > 1)
        writers = {}
        for i in map(str, range(race_keys)):
            writers[res[0]["got"][i]] = writers.get(res[0]["got"][i], 0) + 1
        out["race"][str(n)] = {"requests": race_keys, "processes": n, "requests_with_disagreement": disagree,
                               "conflicts_replaced": sum(r["conflicts"] for r in res),
                               "forward_passes": sum(r["computed"] for r in res),
                               "decisions_by_winning_process": sorted(writers.values(), reverse=True)}
        # A steady mix: 90% repeats of a warm set, 10% new requests.
        path = os.path.join(tmp, "mix_%d.sqlite" % n)
        warm = DecisionCache(path)
        drv = Driver(warm, base_payload())
        for i in range(keys):
            drv.request("mix %d" % i, QUESTION_SETS["billing"])
        warm.close()
        res = launch(n, path=path, mode="mix", keys=keys, seconds=seconds, hit_share=0.9)
        lat = [x for r in res for x in r["lat"]]
        mixed = out["mixed_90pct_hits"][str(n)] = {"requests_per_s": round(len(lat) / seconds), **dist_us(lat)}
        print("  %d processes: race disagreements %d, conflicts %d | mixed %d req/s, p99 %.1f us, max %.1f us"
              % (n, disagree, out["race"][str(n)]["conflicts_replaced"], mixed["requests_per_s"], mixed["p99_us"],
                 mixed["max_us"]), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sections", default="tail,maintenance,renewal,bytes,threads,processes")
    parser.add_argument("--sizes", default="10000,100000,1000000")
    parser.add_argument("--ops", type=int, default=20_000, help="timed calls per store and size in `tail`")
    parser.add_argument("--out")
    args = parser.parse_args()
    sections = [s for s in args.sections.split(",") if s]
    payload = base_payload()
    blob = DecisionCache()._encode(payload)
    results = {"meta": {"veya": veya.__version__, "python": platform.python_version(), "machine": platform.machine(),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "platform": platform.platform(), "cpus": os.cpu_count(), "sqlite": sqlite3.sqlite_version,
                        "load_average_at_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
                        "parameters": {"sections": sections, "sizes": args.sizes, "ops": args.ops},
                        "source_sha256": {name: hashlib.sha256(Path(os.path.join(ROOT, name)).read_bytes()).hexdigest()
                            for name in ("veya/consistency.py", "research/scripts/bench_store_scaling.py",
                                         "research/scripts/bench_retention.py",
                                         "research/benchmarks/feishu_zh/results/v1/laya/raw.jsonl")}}}
    with tempfile.TemporaryDirectory() as tmp:
        if "tail" in sections:
            print("tail", flush=True)
            results["tail"] = tail(tmp, blob, [int(x) for x in args.sizes.split(",")], args.ops)
        if "maintenance" in sections:
            print("maintenance", flush=True)
            results["maintenance"] = maintenance(tmp, blob, [int(x) for x in args.sizes.split(",")])
        if "renewal" in sections:
            print("renewal", flush=True)
            results["renewal"] = renewal(tmp, payload)
            print(json.dumps(results["renewal"], indent=1))
        if "bytes" in sections:
            print("bytes", flush=True)
            results["bytes"] = bytes_held(tmp, payloads())
            print(json.dumps(results["bytes"], indent=1))
        if "threads" in sections:
            print("threads", flush=True)
            results["threads"] = threads(tmp, payload)
        if "processes" in sections:
            print("processes", flush=True)
            results["processes"] = processes(tmp)
    results["meta"]["load_average_at_end"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
