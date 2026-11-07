"""How often a Laya answer changes with the setup around the model, on the real checkpoints.

Every request of a request_sets.py JSONL is answered by each setup below, and each setup is
compared with PyTorch fp32 on CPU answering the request on its own:

    cpu-fp32            the reference; run twice, to check it repeats byte for byte
    cpu-fp32-batched    the same request inside a batch of five (padding to the longest state)
    cpu-int8            the encoder's nn.Linear layers dynamically quantized to int8
    mps-default         Apple silicon with Laya's default policy: fp32 below mps_amp_min_rows rows,
                        fp16 autocast from there (default 5); run twice
    mps-default-batched the same request in a batch of five on the default policy, which puts it
                        over the fp16 threshold
    mps-fp32, mps-fp16  Apple silicon in fp32 always, and in fp16 autocast always
    onnx-LABEL          ONNXAgent on each --onnx LABEL=PATH (English checkpoint only)

Per setup it reports the answers with the same decision (argmax, or P(yes) >= 0.5), requests
whose `answers` are byte-identical (the payload's `model` field names the backend, so it is left
out), the largest probability change, the reference margin (`decision_margins`) of every changed
decision, how many answers a margin threshold would send to review to catch them all, and
accuracy wherever the set carries gold labels.

Then it checks that a DecisionCache removes the variation: the reference fills one SQLite file,
and every other setup answers through it, replaying instead of computing.

    python research/scripts/request_sets.py --out requests.jsonl
    python scripts/export_onnx.py --model convaiinnovations/laya --output english.onnx --int8
    python research/scripts/bench_backend_consistency.py --requests requests.jsonl \\
        --onnx fp32=english.onnx --onnx int8=english.int8.onnx --out consistency_m4.json
"""
import argparse
import json
import os
import platform
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
from veya import DecisionCache, Router, decision_margins  # noqa: E402
from veya.agent import Agent  # noqa: E402
from veya.router import _split  # noqa: E402
from bench_cpu_fast_path import decision, encoder_int8, onnx_agent, probabilities  # noqa: E402
from request_sets import load  # noqa: E402

if torch.backends.quantized.engine == "none" and "qnnpack" in torch.backends.quantized.supported_engines:
    # Apple silicon builds ship qnnpack only and select no engine, so int8 would not load at all.
    torch.backends.quantized.engine = "qnnpack"

BATCH_COMPANY = 4  # other states sharing the batch in the *-batched setups
THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2)


def canonical(result):
    return json.dumps(result, sort_keys=True, ensure_ascii=False)


def answer_all(agent, requests, batched=False, hooks=None):
    out = []
    for i, r in enumerate(requests):
        if batched:
            # Fixed company per request: the next states of the same set, whatever their length.
            company = [requests[(i + k) % len(requests)]["state"] for k in range(1, BATCH_COMPANY + 1)]
            out.append(agent.predict_batch([r["state"]] + company, r["questions"], hooks=hooks)[0])
        else:
            out.append(agent.predict(r["state"], r["questions"], hooks=hooks))
    return out


def compare(requests, ref, other):
    same = identical = total = 0
    diffs, changed, ref_margins = [], [], []
    for r, a, b in zip(requests, ref, other):
        identical += canonical(a["answers"]) == canonical(b["answers"])  # `model` names the backend
        margins = decision_margins(a)
        for q, x in a["answers"].items():
            y = b["answers"][q]
            px, py = probabilities(x), probabilities(y)
            diffs.extend(abs(px[k] - py[k]) for k in px)
            m = margins.get(q, 1.0)
            ref_margins.append(m)
            total += 1
            if decision(x) == decision(y):
                same += 1
            else:
                changed.append({"request": r["id"], "question": q, "ref_margin": m,
                                "ref": decision(x), "other": decision(y)})
    worst = max((c["ref_margin"] for c in changed), default=None)
    review = {str(t): {"answers_below": sum(m < t for m in ref_margins),
                       "changed_caught": sum(c["ref_margin"] < t for c in changed)} for t in THRESHOLDS}
    return {"requests": len(requests), "identical_payloads": identical, "answers": total,
            "same_decision": same, "changed_decisions": len(changed),
            "changed_share": round(len(changed) / total, 5) if total else None,
            "max_abs_prob_diff": round(max(diffs), 6) if diffs else 0.0,
            "mean_abs_prob_diff": round(statistics.mean(diffs), 7) if diffs else 0.0,
            "changed_ref_margin_max": worst, "review_at_threshold": review, "changed": changed}


def accuracy(requests, results):
    right = n = 0
    for r, res in zip(requests, results):
        for q, label in (r.get("gold") or {}).items():
            n += 1
            right += decision(res["answers"][q]) == label
    return {"right": right, "n": n, "accuracy": round(right / n, 4) if n else None}


def mps_policy(agent, mode):
    """Set an MPS agent's autocast policy: 'default', 'fp32' or 'fp16'."""
    from veya.agent import _mps_amp_min_rows

    agent.dtype = torch.float16
    agent.amp_enabled = mode != "fp32"
    agent.mps_amp_min_rows = _mps_amp_min_rows() if mode == "default" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--requests", required=True)
    parser.add_argument("--onnx", action="append", default=[], metavar="LABEL=PATH",
                        help="ONNX exports of the English checkpoint")
    parser.add_argument("--models", default="english,multilingual")
    parser.add_argument("--skip", default="", help="comma-separated setups to leave out")
    parser.add_argument("--limit", type=int, help="first N requests per model (smoke tests)")
    parser.add_argument("--out")
    args = parser.parse_args()
    onnx_paths = dict(item.split("=", 1) for item in args.onnx)
    skip = {s for s in args.skip.split(",") if s}
    mps = torch.backends.mps.is_available()

    requests = load(args.requests)
    router = Router(device="cpu")
    by_model = {}
    for r in requests:
        by_model.setdefault(router.route(r["state"], r["questions"]).model, []).append(r)
    report = {"meta": {"laya": veya.__version__, "torch": torch.__version__, "platform": platform.platform(),
                       "machine": platform.machine(), "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                       "mps": mps, "requests": os.path.basename(args.requests), "batch_company": BATCH_COMPANY,
                       "load_average": os.getloadavg(),
                       "routed": {m: len(v) for m, v in by_model.items()}},
              "models": {}}
    print(json.dumps(report["meta"], indent=2), flush=True)

    for model in [m for m in args.models.split(",") if m in by_model]:
        reqs = by_model[model][:args.limit] if args.limit else by_model[model]
        repo, sub = _split(router.models[model])
        setups, times = {}, {}

        def run(name, agent, **kw):
            if name in skip:
                return
            t = time.perf_counter()
            setups[name] = answer_all(agent, reqs, **kw)
            times[name] = round(time.perf_counter() - t, 1)
            print("  %-14s %-20s %7.1f s" % (model, name, times[name]), flush=True)

        cpu = Agent(repo, subfolder=sub, device="cpu")
        run("cpu-fp32", cpu)
        run("cpu-fp32-again", cpu)
        run("cpu-fp32-batched", cpu, batched=True)
        if "cpu-int8" not in skip:
            run("cpu-int8", encoder_int8(cpu))
        onnx = {}
        if model == "english":
            for label, path in onnx_paths.items():
                onnx["onnx-" + label] = onnx_agent(router.models[model], path, None)
                run("onnx-" + label, onnx["onnx-" + label])
        gpu = None
        if mps:
            gpu = Agent(repo, subfolder=sub, device="mps")
            mps_policy(gpu, "default")
            run("mps-default", gpu)
            run("mps-default-again", gpu)
            run("mps-default-batched", gpu, batched=True)
            mps_policy(gpu, "fp32")
            run("mps-fp32", gpu)
            mps_policy(gpu, "fp16")
            run("mps-fp16", gpu)
            mps_policy(gpu, "default")

        ref = setups["cpu-fp32"]
        entry = {"requests": len(reqs), "seconds": times, "accuracy": {}, "vs_cpu_fp32": {}}
        for name, results in setups.items():
            entry["accuracy"][name] = accuracy(reqs, results)
            if name != "cpu-fp32":
                entry["vs_cpu_fp32"][name] = compare(reqs, ref, results)
        # Per request set, so the sets are not blended into one share.
        entry["by_set"] = {}
        for set_name in sorted({r["set"] for r in reqs}):
            idx = [i for i, r in enumerate(reqs) if r["set"] == set_name]
            sub_reqs = [reqs[i] for i in idx]
            entry["by_set"][set_name] = {
                name: {"changed": compare(sub_reqs, [ref[i] for i in idx], [res[i] for i in idx])["changed_decisions"],
                       "answers": sum(len(reqs[i]["questions"]) for i in idx),
                       "accuracy": accuracy(sub_reqs, [res[i] for i in idx])["accuracy"]}
                for name, res in setups.items()}

        # One cache in front of every setup: the reference stores, every other setup replays.
        with tempfile.TemporaryDirectory() as tmp:
            cache = DecisionCache(os.path.join(tmp, "decisions.sqlite"))
            t = time.perf_counter()
            stored = answer_all(cpu, reqs, hooks=[cache])
            fill_s = time.perf_counter() - t
            replay = {"fill_s": round(fill_s, 2), "setups": {}}
            others = [("cpu-int8", encoder_int8(cpu), False)] + [(n, a, False) for n, a in onnx.items()]
            if gpu is not None:
                others += [("mps-default", gpu, False), ("mps-default-batched", gpu, True)]
            for name, agent, batched in others:
                before = cache.cache_info()
                t = time.perf_counter()
                got = answer_all(agent, reqs, batched=batched, hooks=[cache])
                seconds = time.perf_counter() - t
                after = cache.cache_info()
                target = [canonical(x) for x in stored]
                replay["setups"][name] = {
                    "identical_to_first_decision": sum(canonical(x) == y for x, y in zip(got, target)),
                    "requests": len(reqs),
                    # In a batch, the other states are requests of their own that were never stored.
                    ("misses_incl_batch_company" if batched else "misses"): after["misses"] - before["misses"],
                    "seconds": round(seconds, 3), "ms_per_request": round(seconds / len(reqs) * 1e3, 3)}
            cache.close()
        entry["decision_cache_replay"] = replay
        report["models"][model] = entry

        print("\n%s: %d requests, %d answers" % (model, len(reqs), sum(len(r["questions"]) for r in reqs)))
        print("  %-20s %11s %9s %11s %12s  %s" % ("setup", "same bytes", "changed", "max |dp|", "accuracy",
                                                 "ref margins of changed"))
        for name in setups:
            c = entry["vs_cpu_fp32"].get(name)
            acc = entry["accuracy"][name]["accuracy"]
            print("  %-20s %11s %9s %11s %12s  %s" % (
                name, "%d/%d" % (c["identical_payloads"], c["requests"]) if c else "-",
                c["changed_decisions"] if c else "-", c["max_abs_prob_diff"] if c else "-", acc,
                ", ".join("%.4f" % x["ref_margin"] for x in (c["changed"] if c else [])[:10])))
        for name, r in replay["setups"].items():
            print("  cache replay %-20s identical %d/%d, misses %s, %.3f ms/request"
                  % (name, r["identical_to_first_decision"], r["requests"],
                     r.get("misses", "%s incl. batch company" % r.get("misses_incl_batch_company")),
                     r["ms_per_request"]))
        del cpu, gpu, onnx
        if mps:
            torch.mps.empty_cache()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
