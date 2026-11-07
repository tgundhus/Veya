"""Two checkouts of Laya on the same requests: are the answers identical, and which is faster?

Each checkout runs in its own subprocess, importing `laya` from its own tree, and the two
alternate for --rounds rounds so drift in the machine cancels out. Every round a worker:

1. answers every request of a request_sets.py JSONL through a `Router` on --device, and keeps
   the whole payload (routing included) as canonical JSON, to compare byte for byte;
2. times the bench_stages.py scenarios as new requests: the Router's memory of recent language
   detections, where there is one, is cleared before every call;
3. times the work around the model on its own: routing, and tokenizing the state and the
   question rows (`Agent._encode_state`), which are what the branch's speed-ups changed.

    python research/scripts/request_sets.py --out requests.jsonl
    git worktree add ../laya-main main
    python research/scripts/bench_branch_ab.py --base ../laya-main --requests requests.jsonl \\
        --device cpu --out ab_cpu.json

`--head` defaults to this checkout. No DecisionCache is installed anywhere: this compares the
default paths, which the branch must leave unchanged. Latency intervals come two ways: samples
resampled one by one, and whole rounds resampled, which is the honest one on a busy machine. Run
the base against itself (`--head` = `--base`) to see the noise floor, and `--reuse DIR` to summarize
worker results again without rerunning them.
"""
import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def scenarios():
    """The bench_stages.py scenarios, as plain data a worker of either checkout can read."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    from bench_stages import SCENARIOS

    return {name: {"model": m, "state": s, "questions": q} for name, (m, s, q, _) in SCENARIOS.items()}


# ---------------------------------------------------------------------------------------- worker

def worker(job):
    started = time.perf_counter()
    sys.path.insert(0, job["root"])
    import importlib
    package = "veya" if os.path.isdir(os.path.join(job["root"], "veya")) else "laya"
    veya = importlib.import_module(package)
    import_s = time.perf_counter() - started
    import torch

    router_mod = importlib.import_module(package + ".router")
    Router = veya.Router
    Agent = importlib.import_module(package + ".agent").Agent

    try:
        import psutil
        rss = lambda: round(psutil.Process().memory_info().rss / 2 ** 20, 1)  # noqa: E731
    except ImportError:
        rss = lambda: None  # noqa: E731

    assert os.path.dirname(os.path.dirname(os.path.abspath(veya.__file__))) == os.path.abspath(job["root"]), \
        "imported laya from %s, not %s" % (veya.__file__, job["root"])
    memo = getattr(router_mod, "_DETECTIONS", None)

    def forget():
        if memo is not None:
            memo.clear()

    device = job["device"]
    if device == "cpu" and job.get("threads"):
        torch.set_num_threads(job["threads"])
    router = Router(device=device)
    agents = {name: router.load(name) for name in ("english", "multilingual")}
    out = {"meta": {"root": job["root"], "laya": veya.__version__, "import_laya_s": round(import_s, 3),
                    "torch": torch.__version__, "device": str(next(iter(agents.values())).device),
                    "torch_threads": torch.get_num_threads(), "rss_mb_loaded": rss()}}

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    # 1. Answers, whole payload.
    requests = [json.loads(line) for line in open(job["requests"], encoding="utf-8") if line.strip()]
    answers = {}
    t0 = time.perf_counter()
    for r in requests if job["answers"] else requests[:20]:  # a few still warm the kernels
        forget()
        answers[r["id"]] = json.dumps(router.predict(r["state"], r["questions"]), sort_keys=True,
                                      ensure_ascii=False)
    out["answers"] = answers if job["answers"] else None
    out["answers_s"] = round(time.perf_counter() - t0, 2)

    # 2. End-to-end latency of new requests.
    latency = {}
    for name, sc in job["scenarios"].items():
        state, questions = sc["state"], sc["questions"]

        def call():
            forget()
            if isinstance(state, list):
                router.predict_batch([{"state": s, "questions": questions} for s in state])
            else:
                router.predict(state, questions)
            sync()
        for _ in range(job["warmup"]):
            call()
        samples = []
        for _ in range(job["repeats"]):
            t = time.perf_counter()
            call()
            samples.append(time.perf_counter() - t)
        latency[name] = samples
    out["latency_s"] = latency

    # 3. The work around the model: routing, and tokenizing the state and question rows.
    prep = {}
    for name, sc in job["scenarios"].items():
        states = sc["state"] if isinstance(sc["state"], list) else [sc["state"]]
        questions = sc["questions"]
        agent = agents[sc["model"]]
        ids = list(questions)
        internal = {q: Agent._to_internal(questions[q]) for q in ids}

        def route():
            for s in states:
                forget()
                router.route(s, questions)

        def encode():
            for s in states:
                agent._encode_state(s, ids, internal)
        for label, fn in (("routing", route), ("tokenize state and rows", encode)):
            for _ in range(3):
                fn()
            n = max(3, min(200, int(0.25 / max(1e-6, _once(fn)))))
            runs = []
            for _ in range(7):
                t = time.perf_counter()
                for _ in range(n):
                    fn()
                runs.append((time.perf_counter() - t) / n)
            prep.setdefault(name, {})[label] = runs
    out["prep_s"] = prep
    out["meta"]["rss_mb_end"] = rss()
    return out


def _once(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


# ---------------------------------------------------------------------------------------- driver

def med_ms(xs):
    return round(statistics.median(xs) * 1e3, 3)


def ratio_ci(base, head, n=2000, seed=0):
    """Median(base) / median(head) with a 95% bootstrap interval: above 1 means head is faster."""
    rng = random.Random(seed)
    point = statistics.median(base) / statistics.median(head)
    boots = sorted(statistics.median(rng.choices(base, k=len(base))) / statistics.median(rng.choices(head, k=len(head)))
                   for _ in range(n))
    return [round(point, 4), round(boots[int(0.025 * n)], 4), round(boots[int(0.975 * n)], 4)]


def ratio_ci_rounds(base_rounds, head_rounds, n=2000, seed=0):
    """The same ratio with rounds resampled instead of samples.

    Samples within one process share its load and state, so resampling them one by one gives an
    interval that is too narrow; an A/A run of one checkout against itself shows it. Resampling
    whole rounds (their medians) is the honest interval.
    """
    rng = random.Random(seed)
    b = [statistics.median(r) for r in base_rounds]
    h = [statistics.median(r) for r in head_rounds]
    point = statistics.median(b) / statistics.median(h)
    boots = sorted(statistics.median(rng.choices(b, k=len(b))) / statistics.median(rng.choices(h, k=len(h)))
                   for _ in range(n))
    return [round(point, 4), round(boots[int(0.025 * n)], 4), round(boots[int(0.975 * n)], 4)]


def compare_answers(a, b):
    """Byte-identical count, and for the rest the largest probability gap and changed decisions."""
    sys.path.insert(0, HERE)
    from bench_cpu_fast_path import decision, probabilities

    same, differ = 0, []
    for rid, ja in a.items():
        jb = b[rid]
        if ja == jb:
            same += 1
            continue
        ra, rb = json.loads(ja), json.loads(jb)
        worst, flips = 0.0, []
        for q, x in ra.get("answers", {}).items():
            y = rb["answers"][q]
            px, py = probabilities(x), probabilities(y)
            worst = max([worst] + [abs(px[k] - py[k]) for k in px])
            if decision(x) != decision(y):
                flips.append(q)
        differ.append({"request": rid, "max_abs_prob_diff": round(worst, 6), "changed_decisions": flips,
                       "routing_same": ra.get("routing") == rb.get("routing")})
    return {"identical": same, "requests": len(a), "differ": differ}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="checkout to compare against, e.g. a worktree of main")
    parser.add_argument("--head", default=ROOT, help="checkout under test; default this one")
    parser.add_argument("--requests", required=True, help="a request_sets.py JSONL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--answer-rounds", type=int, default=2,
                        help="rounds that answer the whole request set; the rest only time")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--scenarios", default="all", help="semicolon-separated bench_stages.py scenario names")
    parser.add_argument("--reuse", metavar="DIR",
                        help="summarize the worker results already in DIR instead of running (same --rounds)")
    parser.add_argument("--out")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        job = json.load(open(args.worker, encoding="utf-8"))
        result = worker(job)
        with open(job["result"], "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        return

    trees = {"base": os.path.abspath(args.base), "head": os.path.abspath(args.head)}
    runs = {"base": [], "head": []}
    tmp = args.reuse or tempfile.mkdtemp(prefix="laya_ab_")
    sc = scenarios()
    if args.scenarios != "all":
        sc = {name: sc[name] for name in (x.strip() for x in args.scenarios.split(";"))}
    for rnd in range(args.rounds if args.reuse is None else 0):
        order = ("base", "head") if rnd % 2 == 0 else ("head", "base")
        for which in order:
            job = {"root": trees[which], "requests": os.path.abspath(args.requests), "device": args.device,
                   "threads": args.threads, "scenarios": sc, "repeats": args.repeats, "warmup": args.warmup,
                   "answers": rnd < args.answer_rounds,
                   "result": os.path.join(tmp, "%s_%d.json" % (which, rnd))}
            path = os.path.join(tmp, "job_%s_%d.json" % (which, rnd))
            json.dump(job, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            t = time.perf_counter()
            # -I: nothing from the environment or the working directory decides which laya loads.
            subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--base", args.base,
                            "--requests", args.requests, "--worker", path], check=True, cwd=tmp)
            runs[which].append(json.load(open(job["result"], encoding="utf-8")))
            print("round %d %-4s done in %.0f s" % (rnd + 1, which, time.perf_counter() - t), flush=True)
    if args.reuse:
        for which in runs:
            runs[which] = [json.load(open(os.path.join(tmp, "%s_%d.json" % (which, rnd)), encoding="utf-8"))
                           for rnd in range(args.rounds)]
        sc = {name: sc[name] for name in runs["base"][0]["latency_s"]}

    base, head = runs["base"], runs["head"]
    report = {"meta": {"base": base[0]["meta"], "head": head[0]["meta"], "device": args.device,
                       "platform": platform.platform(), "machine": platform.machine(),
                       "load_average_at_end": os.getloadavg(),
                       "rounds": args.rounds, "repeats": args.repeats, "warmup": args.warmup,
                       "requests": os.path.basename(args.requests)}}
    # Answers: the same between the checkouts, and the same across rounds within each.
    if base[0]["answers"] is not None:
        report["answers_base_vs_head"] = compare_answers(base[0]["answers"], head[0]["answers"])
    report["answers_repeatable_across_rounds"] = {
        w: all(r["answers"] == rs[0]["answers"] for r in rs[1:] if r["answers"] is not None)
        for w, rs in runs.items()}
    report["answer_runs_per_checkout"] = min(args.answer_rounds, args.rounds)
    report["latency"] = {}
    for name in sc:
        b = [x for r in base for x in r["latency_s"][name]]
        h = [x for r in head for x in r["latency_s"][name]]
        report["latency"][name] = {"base_median_ms": med_ms(b), "head_median_ms": med_ms(h),
                                   "saved_ms": round(med_ms(b) - med_ms(h), 3),
                                   "speedup_ci95": ratio_ci(b, h),
                                   "speedup_ci95_by_round": ratio_ci_rounds([r["latency_s"][name] for r in base],
                                                                            [r["latency_s"][name] for r in head]),
                                   "samples": len(b)}
    report["prep"] = {}
    for name in sc:
        for label in base[0]["prep_s"][name]:
            b = [x for r in base for x in r["prep_s"][name][label]]
            h = [x for r in head for x in r["prep_s"][name][label]]
            report["prep"].setdefault(name, {})[label] = {
                "base_median_ms": med_ms(b), "head_median_ms": med_ms(h),
                "saved_ms": round(med_ms(b) - med_ms(h), 4), "speedup_ci95": ratio_ci(b, h),
                "speedup_ci95_by_round": ratio_ci_rounds([r["prep_s"][name][label] for r in base],
                                                         [r["prep_s"][name][label] for r in head])}
    report["import_laya_s"] = {w: [r["meta"]["import_laya_s"] for r in rs] for w, rs in runs.items()}
    report["rss_mb"] = {w: [[r["meta"]["rss_mb_loaded"], r["meta"]["rss_mb_end"]] for r in rs]
                        for w, rs in runs.items()}

    a = report.get("answers_base_vs_head")
    if a:
        print("\nanswers identical base vs head: %d/%d; repeatable across rounds: %s"
              % (a["identical"], a["requests"], report["answers_repeatable_across_rounds"]))
    print("\n%-45s %10s %10s %9s  %s" % ("scenario (new request)", "base ms", "head ms", "saved",
                                         "speedup [95% CI, by sample] [by round]"))
    for name, r in report["latency"].items():
        print("%-45s %10.2f %10.2f %9.2f  %s %s" % (name, r["base_median_ms"], r["head_median_ms"], r["saved_ms"],
                                                     r["speedup_ci95"], r["speedup_ci95_by_round"]))
    print("\n%-45s %-24s %9s %9s  %s" % ("scenario", "stage", "base ms", "head ms", "speedup [by sample] [by round]"))
    for name, stages in report["prep"].items():
        for label, r in stages.items():
            print("%-45s %-24s %9.3f %9.3f  %s %s" % (name, label, r["base_median_ms"], r["head_median_ms"],
                                                     r["speedup_ci95"], r["speedup_ci95_by_round"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
