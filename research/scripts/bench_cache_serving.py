"""What a DecisionCache does for a served workload, on the real checkpoints.

Sections:

    trace     a stream of requests drawn from a request_sets.py pool with Zipf popularity, served
              one after another by a Router without a cache (the default path) and with a memory
              and a SQLite cache: requests per second, latency percentiles, and whether every
              repeat got the answer its request got the first time.
    coalesce  K threads send the same new request at once, with coalescing on and off: forward
              passes run, time until all K have an answer, and whether they all got the same bytes.
              On CPU by default: PyTorch's MPS backend is not safe for concurrent forward passes
              from several threads (a Metal assertion aborts the process), so "off" cannot run there.
              `mps_threads` checks that in a subprocess: two threads, two different new requests.
    serve     the HTTP server (veya.serve.create_app, in process over httpx) with a cache in its
              Router: a hit on its own, a miss on its own, and a hit while other clients' misses
              are running. The server runs one prediction at a time behind a lock, hits included.

    python research/scripts/request_sets.py --out requests.jsonl
    python research/scripts/bench_cache_serving.py --requests requests.jsonl --device mps --out serving.json
"""
import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

import veya  # noqa: E402
import veya.router as router_mod  # noqa: E402
from veya import DecisionCache, Router  # noqa: E402
from request_sets import load  # noqa: E402


def pct(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: round(xs[min(n - 1, int(p * n))] * 1e3, 3)  # noqa: E731
    return {"n": n, "median_ms": q(0.5), "p90_ms": q(0.9), "p99_ms": q(0.99), "max_ms": round(xs[-1] * 1e3, 3),
            "mean_ms": round(statistics.mean(xs) * 1e3, 3)}


def canonical(result):
    return json.dumps({k: v for k, v in result.items() if k != "routing"}, sort_keys=True, ensure_ascii=False)


# ----------------------------------------------------------------------------------------- trace

def make_trace(pool, n, s, seed):
    rng = random.Random(seed)
    weights = [1.0 / (k ** s) for k in range(1, len(pool) + 1)]
    order = list(range(len(pool)))
    rng.shuffle(order)                 # popularity unrelated to position in the file
    return [pool[order[i]] for i in rng.choices(range(len(pool)), weights=weights, k=n)]


def serve_trace(router, trace, device):
    lat, first, repeats, same = [], {}, 0, 0
    t_all = time.perf_counter()
    for r in trace:
        if r["id"] not in first:
            router_mod._DETECTIONS.clear()  # a new state is new to routing too
        t = time.perf_counter()
        # No device sync: predict returns host values, so a miss has finished its GPU work, and a
        # hit has none to wait for.
        out = router.predict(r["state"], r["questions"])
        lat.append(time.perf_counter() - t)
        c = canonical(out)
        if r["id"] in first:
            repeats += 1
            same += c == first[r["id"]]
        else:
            first[r["id"]] = c
    total = time.perf_counter() - t_all
    return {"requests": len(trace), "distinct": len(first), "total_s": round(total, 2),
            "requests_per_s": round(len(trace) / total, 2), "latency": pct(lat),
            "repeats": repeats, "repeats_same_answer": same}


def run_trace(pool, device, n, s, tmp):
    router = Router(device=device)
    agents = {m: router.load(m) for m in ("english", "multilingual")}
    trace = make_trace(pool, n, s, seed=11)
    # Warm every request's kernels and caches once, outside the timed runs, so the first
    # setup timed does not pay for them.
    for r in pool[:20]:
        router.predict(r["state"], r["questions"])
    out = {"pool": len(pool), "requests": n, "zipf_s": s, "distinct_in_trace": len({r["id"] for r in trace})}
    setups = [("no cache (default path)", []), ("memory DecisionCache", [DecisionCache()]),
              ("SQLite DecisionCache", [DecisionCache(os.path.join(tmp, "trace.sqlite"))])]
    for name, hooks in setups:
        r = Router(device=device, hooks=hooks)
        for m, agent in agents.items():
            r.attach(m, agent)
        res = serve_trace(r, trace, device)
        if hooks:
            res["cache_info"] = hooks[0].cache_info()
        out[name] = res
        print("  trace %-26s %7.2f req/s  median %8.2f ms  p99 %8.2f ms  repeats same %d/%d"
              % (name, res["requests_per_s"], res["latency"]["median_ms"], res["latency"]["p99_ms"],
                 res["repeats_same_answer"], res["repeats"]), flush=True)
    return out


# -------------------------------------------------------------------------------------- coalesce

MPS_THREADS = r"""
import sys, threading
sys.path.insert(0, %(root)r)
from veya import Router
router = Router(device="mps")
agent = router.load("english")
q = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
router.predict("warm up", q)
def work(i):
    for n in range(20):
        router.predict("thread %%d request %%d: we were billed twice" %% (i, n), q)
ts = [threading.Thread(target=work, args=(i,)) for i in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
print("completed")
"""


def run_mps_threads():
    import subprocess

    done = subprocess.run([sys.executable, "-c", MPS_THREADS % {"root": ROOT}], capture_output=True, text=True)
    tail = [line for line in (done.stderr or "").splitlines() if "assertion" in line.lower()]
    return {"returncode": done.returncode, "completed": "completed" in done.stdout,
            "assertion": tail[-1][-160:] if tail else None}


def run_coalesce(pool, device, threads=(2, 4, 8, 16)):
    router = Router(device=device)
    agents = {m: router.load(m) for m in ("english", "multilingual")}
    calls = {"n": 0}
    count_lock = threading.Lock()  # `+=` is not atomic, and up to 16 threads run passes at once
    for agent in agents.values():
        inner = agent._forward

        def counted(b, inner=inner):
            with count_lock:
                calls["n"] += 1
            return inner(b)
        agent._forward = counted
    req = next(r for r in pool if r["set"] == "english13" and len(r["questions"]) == 3)
    router.predict(req["state"], req["questions"])  # warm
    out = {"request": req["id"]}
    for k in threads:
        for coalesce in (True, False):
            cache = DecisionCache(coalesce=coalesce)
            r = Router(device=device, hooks=[cache])
            for m, a in agents.items():
                r.attach(m, a)
            state = req["state"] + " (%d %s)" % (k, coalesce)  # new to the cache every time
            barrier = threading.Barrier(k)
            got = [None] * k
            done_at = [0.0] * k

            def work(i):
                barrier.wait()
                got[i] = canonical(r.predict(state, req["questions"]))
                done_at[i] = time.perf_counter()
            before = calls["n"]
            pool_ = [threading.Thread(target=work, args=(i,)) for i in range(k)]
            t0 = time.perf_counter()
            for th in pool_:
                th.start()
            for th in pool_:
                th.join()
            out.setdefault(str(k), {})["coalesce=%s" % coalesce] = {
                "forward_passes": calls["n"] - before, "all_done_s": round(max(done_at) - t0, 3),
                "identical_answers": len(set(got)) == 1, "cache_info": cache.cache_info()}
        c, n = out[str(k)]["coalesce=True"], out[str(k)]["coalesce=False"]
        print("  coalesce %2d threads: on %d passes %.2f s | off %d passes %.2f s | identical %s/%s"
              % (k, c["forward_passes"], c["all_done_s"], n["forward_passes"], n["all_done_s"],
                 c["identical_answers"], n["identical_answers"]), flush=True)
    return out


# ----------------------------------------------------------------------------------------- serve

async def _serve(app, pool, hits_n=100, misses_n=12, clients=3):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://laya") as client:
        hot = next(r for r in pool if r["set"] == "english13" and len(r["questions"]) == 3)
        body = {"state": hot["state"], "questions": hot["questions"]}
        (await client.post("/v1/systemone", json=body)).raise_for_status()   # stored

        async def timed(b):
            t = time.perf_counter()
            resp = await client.post("/v1/systemone", json=b)
            resp.raise_for_status()
            return time.perf_counter() - t, float(resp.headers.get("X-Inference-Time-Ms", "nan"))

        out = {}
        alone = [await timed(body) for _ in range(hits_n)]
        out["hit alone"] = {"http": pct([a for a, _ in alone]),
                            "inference_ms_median": statistics.median(b for _, b in alone)}
        fresh = [{"state": hot["state"] + " #%d" % i, "questions": hot["questions"]} for i in range(10_000)]
        it = iter(fresh)
        misses = [await timed(next(it)) for _ in range(misses_n)]
        out["miss alone"] = {"http": pct([a for a, _ in misses])}

        stop = asyncio.Event()

        async def missing_client():
            while not stop.is_set():
                await timed(next(it))
        workers = [asyncio.create_task(missing_client()) for _ in range(clients)]
        await asyncio.sleep(0.2)
        mixed = [await timed(body) for _ in range(max(20, hits_n // 4))]
        stop.set()
        await asyncio.gather(*workers)
        out["hit while %d clients send misses" % clients] = {"http": pct([a for a, _ in mixed])}
        return out


def run_serve(pool, device):
    from veya.serve import create_app

    router = Router(device=device, hooks=[DecisionCache()])
    for m in ("english", "multilingual"):
        router.load(m)
    out = asyncio.run(_serve(create_app(router=router), pool))
    for name, r in out.items():
        print("  serve %-36s median %8.2f ms  p99 %8.2f ms" % (name, r["http"]["median_ms"], r["http"]["p99_ms"]),
              flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--requests", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sections", default="trace,coalesce,serve,mps_threads")
    parser.add_argument("--coalesce-device", default="cpu")
    parser.add_argument("--trace-requests", type=int, default=600)
    parser.add_argument("--zipf", type=float, default=1.0)
    parser.add_argument("--pool", default="english13,massive_en,feishu",
                        help="request sets the trace draws from")
    parser.add_argument("--out")
    args = parser.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    pool = load(args.requests, set(args.pool.split(",")))
    results = {"meta": {"laya": veya.__version__, "torch": torch.__version__, "device": device,
                        "platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
                        "torch_threads": torch.get_num_threads(), "load_average_at_start": os.getloadavg()}}
    with tempfile.TemporaryDirectory() as tmp:
        if "trace" in args.sections:
            results["trace"] = run_trace(pool, device, args.trace_requests, args.zipf, tmp)
        if "coalesce" in args.sections:
            results["coalesce"] = run_coalesce(pool, args.coalesce_device)
        if "mps_threads" in args.sections and torch.backends.mps.is_available():
            results["mps_threads"] = run_mps_threads()
            print("  two threads predicting on MPS at once: %s" % results["mps_threads"], flush=True)
        if "serve" in args.sections:
            results["serve"] = run_serve(pool, device)
    results["meta"]["load_average_at_end"] = os.getloadavg()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
