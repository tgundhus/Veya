"""CPU fast-path backends against PyTorch fp32: latency per request and answer agreement.

Backends, each timed on the same requests in round-robin order within every repeat (the order
rotates per repeat, so drift in the machine cancels out):

    torch-fp32   the stock Agent on CPU, the baseline
    torch-int8   the same Agent with the encoder's nn.Linear layers dynamically quantized to int8
                 (the head's nn.TransformerEncoderLayer is left alone: quantizing it breaks the
                 fast-path check in its forward)
    onnx-LABEL   ONNXAgent on each --onnx LABEL=PATH, as written by
                 `python scripts/export_onnx.py --model ID --output english.onnx --int8`
                 (english.onnx for fp32, english.int8.onnx for int8)

For every backend it reports the median latency per request, the speedup against torch-fp32, and
how far its answers are from torch-fp32's: the share with the same decision (argmax for choice
and score, P(yes) >= 0.5 for noul), the max and mean absolute probability difference, and for
every changed decision its torch-fp32 margin from `veya.decision_margins` (a change near the
boundary is a coin flip; one far from it is a real regression).

Published checkpoints (downloads on first use; the ONNX files must match the checkpoint):

    python research/scripts/bench_cpu_fast_path.py --out fastpath_m4.json
    python research/scripts/bench_cpu_fast_path.py --onnx fp32=english.onnx \\
        --onnx int8=english.int8.onnx --out fastpath_cpu.json

Local checkpoints, and the 64 Chinese feishu_zh cases (both modes) on the multilingual model:

    python research/scripts/bench_cpu_fast_path.py --checkpoint english=/path/english \\
        --onnx fp32=/path/english.onnx --onnx int8=/path/english.int8.onnx --out en.json
    python research/scripts/bench_cpu_fast_path.py --feishu --checkpoint multilingual=/path/ml \\
        --onnx fp32=/path/ml.onnx --onnx int8=/path/ml.int8.onnx --out feishu.json

Custom requests: --requests FILE.jsonl with one {"state": ..., "questions": ...} per line
(an optional "id" names the row). Needs torch; onnxruntime only for the onnx backends.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import veya  # noqa: E402
from veya import Router, decision_margins  # noqa: E402
from veya.agent import Agent  # noqa: E402
from veya.router import _split  # noqa: E402
from bench_stages import EMAIL, ONE, TEN, TICKET, TRIAGE  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:  # torch-only runs
    ort = None

FEISHU = os.path.join(ROOT, "research", "benchmarks", "feishu_zh")

SHIPPING = ("Where is my order #A-55120? It was supposed to arrive last Friday and the tracking page has "
            "shown 'label created' for nine days. I need it for my daughter's birthday on Saturday.")
REVIEW = ("Honestly the best headphones I've owned. Battery lasts all week, the noise cancelling is great on "
          "the train, and support replaced a faulty ear cushion within two days. Five stars.")
SECURITY = ("URGENT: we received an alert that someone logged into our admin account from an IP in another "
            "country at 3am. We did not do this. Please lock the account and tell us what was accessed.")
FEATURE = ("Would be nice if the dashboard could export reports as CSV as well as PDF. Not a big deal, we "
           "copy the numbers by hand for now, but our accountant keeps asking.")
PROFILE_CHAT = {"customer": {"plan": "enterprise", "seats": 250, "renewal": "2026-11-01"},
                "messages": [{"from": "customer", "text": "Our SSO integration stopped working after your update."},
                             {"from": "agent", "text": "Sorry about that, can you share the error code?"},
                             {"from": "customer", "text": "SAML_403. Half the company is locked out. If this is "
                                                          "not fixed today we will escalate to our account manager."}]}
SPAM = "Congratulations!!! You have been selected to receive a $500 gift card. Click the link to claim now."
MODERATION = {
    "toxic": {"type": "noul", "instructions": "Is the message abusive, harassing or hateful?"},
    "spam": {"type": "noul", "instructions": "Is this unsolicited advertising or a scam?"},
    "action": {"type": "choice", "instructions": "What should a moderator do?",
               "criteria": {"approve": "harmless, leave it up", "review": "unclear, a human should look",
                            "remove": "clearly violates the rules"}},
}
SUPPORT = {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"order_status": "where is my order, delivery, tracking",
                            "account_security": "unauthorised access, locked account, passwords",
                            "feature_request": "a new capability or improvement",
                            "praise": "positive feedback", "other": "anything else"}},
    "priority": {"type": "score", "instructions": "How quickly must support respond?",
                 "criteria": ["this week", "today", "within the hour", "immediately"]},
    "needs_human": {"type": "noul", "instructions": "Should a human agent take over this conversation?"},
    "sentiment": {"type": "choice", "instructions": "Overall sentiment?",
                  "criteria": {"positive": "", "neutral": "", "negative": ""}},
}

ENGLISH = [
    ("ticket, one (1)", TICKET, ONE),
    ("ticket, triage (3)", TICKET, TRIAGE),
    ("ticket, ten (10)", TICKET, TEN),
    ("email, triage (3)", EMAIL, TRIAGE),
    ("email, ten (10)", EMAIL, TEN),
    ("shipping, support (4)", SHIPPING, SUPPORT),
    ("review, support (4)", REVIEW, SUPPORT),
    ("security, support (4)", SECURITY, SUPPORT),
    ("feature, support (4)", FEATURE, SUPPORT),
    ("chat dict, support (4)", PROFILE_CHAT, SUPPORT),
    ("chat dict, triage (3)", PROFILE_CHAT, TRIAGE),
    ("spam, moderation (3)", SPAM, MODERATION),
    ("review, moderation (3)", REVIEW, MODERATION),
]


def english_requests():
    return [{"id": i, "state": s, "questions": q} for i, s, q in ENGLISH]


def file_requests(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                out.append({"id": str(row.get("id", n)), "state": row["state"], "questions": row["questions"]})
    return out


def feishu_requests():
    sys.path.insert(0, FEISHU)
    from audit import load_cases  # verifies the frozen dataset and prompt hashes
    from prompts import requests_for

    cases, _ = load_cases()
    return [{"id": "%s/%s" % (c["id"], mode), **req} for c in cases for mode, req in requests_for(c).items()]


def decision(answer):
    if "noul" in answer:
        return answer["noul"] >= 0.5
    if answer.get("type") == "choice" and "choice" in answer:
        return answer["choice"]
    probs = answer["probabilities"]
    return max(probs, key=probs.get)


def probabilities(answer):
    if "noul" in answer:
        return {"true": answer["noul"]}
    return answer["probabilities"]


def agreement(base, other, requests):
    """Compare two backends' results per answer against the torch-fp32 ones in `base`."""
    same = total = 0
    diffs, changed, margins_all = [], [], []
    for req, rb, ro in zip(requests, base, other):
        margins = decision_margins(rb)
        for qid, a in rb["answers"].items():
            b = ro["answers"][qid]
            pa, pb = probabilities(a), probabilities(b)
            diffs.extend(abs(pa[k] - pb[k]) for k in pa)
            margins_all.append(margins.get(qid, 1.0))
            total += 1
            if decision(a) == decision(b):
                same += 1
            else:
                changed.append({"request": req["id"], "question": qid, "fp32_margin": margins.get(qid),
                                "fp32": decision(a), "backend": decision(b)})
    return {
        "same_decision": same, "answers": total, "same_decision_share": round(same / total, 4) if total else None,
        "max_abs_prob_diff": round(max(diffs), 5) if diffs else 0.0,
        "mean_abs_prob_diff": round(statistics.mean(diffs), 5) if diffs else 0.0,
        "changed": changed,
        "changed_fp32_margin_max": max((c["fp32_margin"] or 0) for c in changed) if changed else None,
        "fp32_answers_with_margin_below_0.05": sum(m < 0.05 for m in margins_all),
    }


def encoder_int8(agent):
    """A shallow copy of `agent` whose model has its encoder nn.Linear layers quantized to int8."""
    import copy

    names = {n for n, m in agent.model.named_modules() if n.startswith("encoder.") and isinstance(m, torch.nn.Linear)}
    clone = copy.copy(agent)
    clone.model = torch.ao.quantization.quantize_dynamic(agent.model, names, dtype=torch.qint8)
    return clone


def onnx_agent(spec, path, threads):
    from veya.onnx_agent import ONNXAgent

    repo, sub = _split(spec)
    agent = ONNXAgent(repo, onnx_path=path, subfolder=sub)
    if threads:
        # ORT's own default already uses one intra-op thread per physical core; this pins it.
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        agent.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return agent


def summary(seconds):
    ms = sorted(x * 1e3 for x in seconds)
    return {"median": round(statistics.median(ms), 3), "min": round(ms[0], 3),
            "p90": round(ms[min(len(ms) - 1, int(0.9 * len(ms)))], 3)}


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return round(statistics.geometric_mean(xs), 3) if xs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None,
                        help="checkpoint name (english, multilingual, ...); default english, multilingual with --feishu")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="use a local checkpoint directory for NAME instead of the Hub")
    parser.add_argument("--onnx", action="append", default=[], metavar="LABEL=PATH",
                        help="an .onnx for the checkpoint, timed as backend onnx-LABEL (e.g. fp32=..., int8=...)")
    parser.add_argument("--backends", default="torch-fp32,torch-int8,onnx",
                        help="comma separated: torch-fp32 (always kept), torch-int8, onnx (every --onnx), onnx-LABEL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--requests", metavar="FILE.jsonl", help='one {"state": ..., "questions": ...} per line')
    group.add_argument("--feishu", action="store_true",
                       help="the 64 Chinese research/benchmarks/feishu_zh cases, both modes (128 requests)")
    parser.add_argument("--limit", type=int, help="only the first N requests (smoke tests)")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per request and backend")
    parser.add_argument("--warmup", type=int, default=1, help="untimed passes over every request per backend")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch.set_num_threads and ORT intra_op_num_threads; default: each library's own")
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args()
    args.checkpoint = dict(item.split("=", 1) for item in args.checkpoint)
    args.onnx = dict(item.split("=", 1) for item in args.onnx)
    model = args.model or ("multilingual" if args.feishu else "english")
    wanted = {b.strip() for b in args.backends.split(",") if b.strip()} | {"torch-fp32"}
    if args.onnx and ort is None and ("onnx" in wanted or any(b.startswith("onnx-") for b in wanted)):
        raise SystemExit("--onnx needs onnxruntime: pip install onnxruntime")
    if args.threads:
        torch.set_num_threads(args.threads)

    requests = file_requests(args.requests) if args.requests else feishu_requests() if args.feishu \
        else english_requests()
    if args.limit:
        requests = requests[:args.limit]

    spec = args.checkpoint.get(model) or Router().models[model]
    repo, sub = _split(spec)
    backends = {"torch-fp32": Agent(repo, subfolder=sub, device="cpu")}
    if "torch-int8" in wanted:
        try:
            backends["torch-int8"] = encoder_int8(backends["torch-fp32"])
        except (AttributeError, RuntimeError) as e:  # torch.ao.quantization gone, or no int8 engine
            print("torch-int8 skipped: %s" % e, file=sys.stderr)
    for label, path in args.onnx.items():
        if "onnx" in wanted or "onnx-" + label in wanted:
            backends["onnx-" + label] = onnx_agent(spec, path, args.threads)
    names = list(backends)

    meta = {
        "laya": veya.__version__, "torch": torch.__version__,
        "onnxruntime": getattr(ort, "__version__", None), "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(), "ort_threads": args.threads or "default",
        "model": model, "checkpoint": args.checkpoint.get(model, "hub"), "synthetic_weights": model in args.checkpoint,
        "onnx": args.onnx, "requests": "file:" + args.requests if args.requests else "feishu_zh" if args.feishu
        else "built-in english", "n_requests": len(requests), "repeats": args.repeats, "warmup": args.warmup,
    }
    print(json.dumps(meta, indent=2), flush=True)

    def call(agent, req):
        return agent.predict(req["state"], req["questions"])

    # Warm-up passes double as the answers compared below; every backend is deterministic on CPU.
    outputs = {n: [call(backends[n], r) for r in requests] for n in names}
    for _ in range(args.warmup - 1):
        for n in names:
            for r in requests:
                call(backends[n], r)

    times = {n: [[] for _ in requests] for n in names}
    for rep in range(args.repeats):
        for i, req in enumerate(requests):
            k = (rep + i) % len(names)
            for n in names[k:] + names[:k]:
                start = time.perf_counter()
                call(backends[n], req)
                times[n][i].append(time.perf_counter() - start)
        print("repeat %d/%d done" % (rep + 1, args.repeats), flush=True)

    tokens = [o["usage"]["input_tokens"] for o in outputs["torch-fp32"]]
    base_med = [statistics.median(t) for t in times["torch-fp32"]]
    results = {"meta": meta, "backends": {}}
    for n in names:
        med = [statistics.median(t) for t in times[n]]
        per_request = [{"request": r["id"], "questions": len(r["questions"]), "input_tokens": tokens[i],
                        "latency_ms": summary(times[n][i]), "speedup": round(base_med[i] / med[i], 3)}
                       for i, r in enumerate(requests)]
        entry = {
            "median_of_request_medians_ms": round(statistics.median(med) * 1e3, 3),
            "total_of_request_medians_ms": round(sum(med) * 1e3, 3),
            "speedup_total": round(sum(base_med) / sum(med), 3),
            "speedup_geomean": geomean([p["speedup"] for p in per_request]),
            "per_request": per_request,
        }
        if n != "torch-fp32":
            entry["agreement"] = agreement(outputs["torch-fp32"], outputs[n], requests)
        results["backends"][n] = entry

    print("\n%-12s %10s %10s %8s %8s %11s %9s %9s  %s" % (
        "backend", "median ms", "total ms", "x total", "x geo", "same", "max |dp|", "mean |dp|",
        "fp32 margins of changed"))
    for n, e in results["backends"].items():
        a = e.get("agreement")
        print("%-12s %10.1f %10.1f %8.3f %8.3f %11s %9s %9s  %s" % (
            n, e["median_of_request_medians_ms"], e["total_of_request_medians_ms"], e["speedup_total"],
            e["speedup_geomean"] or 0, "%d/%d" % (a["same_decision"], a["answers"]) if a else "-",
            a["max_abs_prob_diff"] if a else "-", a["mean_abs_prob_diff"] if a else "-",
            ", ".join("%.4f" % (c["fp32_margin"] or 0) for c in a["changed"][:12])
            + (" ..." if len(a["changed"]) > 12 else "") if a else ""))
    print("\nper request, median ms (speedup vs torch-fp32):")
    for i, r in enumerate(requests[:40]):
        print("  %-28s %4d tok  " % (str(r["id"])[:28], tokens[i]) + "  ".join(
            "%s %.1f (x%.2f)" % (n, statistics.median(times[n][i]) * 1e3, base_med[i] / statistics.median(times[n][i]))
            for n in names))
    if len(requests) > 40:
        print("  ... %d more in --out" % (len(requests) - 40))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
