"""Two checkouts' request paths around one model, in one process: the fairest new-request comparison.

bench_branch_ab.py runs each checkout in its own process, which on a busy machine lets load that
differs between processes decide a close comparison. Here the other checkout's `laya` package is
copied under the name `veya_base` and imported beside this one. Both Routers get Agents that share
the same model object -- the forward pass is literally the same module -- and requests alternate
between them in adjacent pairs, base first on even pairs and head first on odd ones, so each pair
sees the same load. What differs is only the code around the model: routing and language
detection, question and state tokenization, answer decoding, hooks.

Reported per scenario: the median of per-pair ratios (base time / head time; above 1 means this
checkout is faster) with a 95% bootstrap interval over pairs, both medians, and whether every
pair's answers were identical.

    git worktree add ../veya-main main
    python research/scripts/bench_branch_same_model.py --base ../veya-main --device cpu --out same_model_cpu.json
"""
import argparse
import copy
import json
import os
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

import veya  # noqa: E402
import veya.router as router_mod  # noqa: E402
from veya import Router  # noqa: E402


def import_base(path):
    """The other checkout's package, importable as `veya_base` next to this `laya`."""
    tmp = tempfile.mkdtemp(prefix="veya_base_")
    shutil.copytree(os.path.join(path, "veya" if os.path.isdir(os.path.join(path, "veya")) else "laya"), os.path.join(tmp, "veya_base"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    sys.path.insert(0, tmp)
    import veya_base
    import veya_base.agent
    import veya_base.router

    return veya_base


def base_agent(agent, base):
    """A shallow copy of this checkout's Agent whose methods are the base checkout's: same model."""
    clone = copy.copy(agent)
    clone.__class__ = base.agent.Agent
    return clone


def ratio_ci(ratios, n=4000, seed=0):
    rng = random.Random(seed)
    point = statistics.median(ratios)
    boots = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
    return [round(point, 4), round(boots[int(0.025 * n)], 4), round(boots[int(0.975 * n)], 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="the other checkout, e.g. a worktree of main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    from bench_branch_ab import scenarios

    base = import_base(os.path.abspath(args.base))
    head_router = Router(device=args.device)
    base_router = base.router.Router(device=args.device)
    for name in ("english", "multilingual"):
        agent = head_router.load(name)
        base_router.attach(name, base_agent(agent, base))
    sync = torch.mps.synchronize if args.device == "mps" else (lambda: None)
    memo = router_mod._DETECTIONS

    def call(router, state, questions):
        memo.clear()                  # a new request for this checkout's routing too
        t = time.perf_counter()
        if isinstance(state, list):
            out = router.predict_batch([{"state": s, "questions": questions} for s in state])
        else:
            out = router.predict(state, questions)
        sync()
        return time.perf_counter() - t, out

    report = {"meta": {"head": veya.__file__, "base": base.__file__, "device": args.device,
                       "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                       "platform": platform.platform(), "pairs": args.pairs, "load_average_at_start": os.getloadavg()},
              "scenarios": {}}
    for name, sc in scenarios().items():
        state, questions = sc["state"], sc["questions"]
        for _ in range(args.warmup):
            call(base_router, state, questions)
            call(head_router, state, questions)
        ratios, tb, th, same = [], [], [], True
        for i in range(args.pairs):
            order = (base_router, head_router) if i % 2 == 0 else (head_router, base_router)
            got = {}
            for router in order:
                got[router is head_router] = call(router, state, questions)
            (b, ob), (h, oh) = got[False], got[True]
            strip = (lambda o: [{k: v for k, v in x.items() if k != "routing"} for x in o]) if isinstance(ob, list) \
                else (lambda o: {k: v for k, v in o.items() if k != "routing"})
            same = same and json.dumps(strip(ob), sort_keys=True) == json.dumps(strip(oh), sort_keys=True)
            ratios.append(b / h)
            tb.append(b)
            th.append(h)
        r = {"base_median_ms": round(statistics.median(tb) * 1e3, 3),
             "head_median_ms": round(statistics.median(th) * 1e3, 3),
             "saved_ms_median_of_pairs": round(statistics.median(x - y for x, y in zip(tb, th)) * 1e3, 3),
             "speedup_median_of_pair_ratios_ci95": ratio_ci(ratios), "answers_identical": same}
        report["scenarios"][name] = r
        print("%-45s base %9.2f ms  head %9.2f ms  saved %7.3f ms  pair ratio %s  identical %s" % (
            name, r["base_median_ms"], r["head_median_ms"], r["saved_ms_median_of_pairs"],
            r["speedup_median_of_pair_ratios_ci95"], same), flush=True)
    report["meta"]["load_average_at_end"] = os.getloadavg()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
