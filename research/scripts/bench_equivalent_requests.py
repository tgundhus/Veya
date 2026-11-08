"""Requests that mean the same but are not byte-identical: how often does the answer change?

A DecisionCache replays exact repeats only. These variants of each request are new keys to it,
so whatever they change is variation the cache cannot remove:

    options reversed    a choice or score question's options in reverse order (answers compared
                        by label, so only a changed decision counts)
    options rotated     the first option moved to the end
    trailing space      one space appended to a string state (or to every string of a dict state)
    whitespace doubled  every single space in string values doubled
    lower-cased         string values lower-cased

Each variant is answered by the same Agent on one device and compared with the original request
on the same device, so hardware and precision play no part. Reported per variant: answers with
a changed decision, the original's margin (`decision_margins`) of each changed answer, and the
largest probability change.

    python research/scripts/request_sets.py --out requests.jsonl
    python research/scripts/bench_equivalent_requests.py --requests requests.jsonl --device mps --out equivalent.json
"""
import argparse
import json
import os
import platform
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

import veya  # noqa: E402
from veya import Router, decision_margins  # noqa: E402
from veya.agent import Agent  # noqa: E402
from veya.router import _split  # noqa: E402
from bench_cpu_fast_path import decision, probabilities  # noqa: E402
from request_sets import load  # noqa: E402


def map_strings(value, fn):
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_strings(v, fn) for v in value]
    return value


def reorder(questions, how):
    out = {}
    for name, q in questions.items():
        q = dict(q)
        crit = q.get("criteria")
        if q["type"] == "choice" and isinstance(crit, dict) and len(crit) > 1:
            items = list(crit.items())
            items = items[::-1] if how == "reverse" else items[1:] + items[:1]
            q["criteria"] = dict(items)
        elif q["type"] == "score" and isinstance(crit, list) and len(crit) > 1:
            # A score's levels are ordinal: reordering them changes the question, so leave it.
            pass
        out[name] = q
    return out


VARIANTS = {
    "options reversed": lambda s, q: (s, reorder(q, "reverse")),
    "options rotated": lambda s, q: (s, reorder(q, "rotate")),
    "trailing space": lambda s, q: (map_strings(s, lambda x: x + " "), q),
    "whitespace doubled": lambda s, q: (map_strings(s, lambda x: x.replace(" ", "  ")), q),
    "lower-cased": lambda s, q: (map_strings(s, str.lower), q),
}


def label(answer):
    """The decided label, comparable across option orders."""
    return decision(answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--requests", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out")
    args = parser.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    requests = load(args.requests)
    router = Router(device="cpu")
    by_model = {}
    for r in requests:
        by_model.setdefault(router.route(r["state"], r["questions"]).model, []).append(r)
    report = {"meta": {"laya": veya.__version__, "torch": torch.__version__, "device": device,
                       "platform": platform.platform(), "requests": os.path.basename(args.requests),
                       "routed": {m: len(v) for m, v in by_model.items()}}, "models": {}}
    for model, reqs in by_model.items():
        repo, sub = _split(router.models[model])
        agent = Agent(repo, subfolder=sub, device=device)
        base = [agent.predict(r["state"], r["questions"]) for r in reqs]
        entry = {"requests": len(reqs), "answers": sum(len(r["questions"]) for r in reqs), "variants": {}}
        for vname, make in VARIANTS.items():
            changed, diffs, compared, skipped = [], [], 0, 0
            for r, b in zip(reqs, base):
                state, questions = make(r["state"], r["questions"])
                # Order-sensitive: dicts compare equal whatever their order, and option order is the point.
                if json.dumps([state, questions]) == json.dumps([r["state"], r["questions"]]):
                    skipped += 1            # nothing to vary in this request
                    continue
                v = agent.predict(state, questions)
                margins = decision_margins(b)
                for q, a in b["answers"].items():
                    x = v["answers"][q]
                    compared += 1
                    pa, px = probabilities(a), probabilities(x)
                    diffs.extend(abs(pa[k] - px[k]) for k in pa if k in px)
                    if label(a) != label(x):
                        changed.append({"request": r["id"], "question": q, "margin": margins.get(q),
                                        "original": label(a), "variant": label(x)})
            ms = sorted(c["margin"] for c in changed if c["margin"] is not None)
            entry["variants"][vname] = {
                "answers_compared": compared, "requests_unchanged_by_variant": skipped,
                "changed_decisions": len(changed),
                "changed_share": round(len(changed) / compared, 4) if compared else None,
                "max_abs_prob_diff": round(max(diffs), 4) if diffs else 0.0,
                "mean_abs_prob_diff": round(statistics.mean(diffs), 5) if diffs else 0.0,
                "changed_margin_median": ms[len(ms) // 2] if ms else None,
                "changed_margin_max": ms[-1] if ms else None,
                "changed_with_margin_below_0.1": sum(m < 0.1 for m in ms),
                "changed": changed[:200]}
            e = entry["variants"][vname]
            print("  %-13s %-19s changed %4d/%-5d (%5.1f%%)  max |dp| %.3f  margin of changed: median %s max %s"
                  % (model, vname, e["changed_decisions"], compared, 100 * (e["changed_share"] or 0),
                     e["max_abs_prob_diff"], e["changed_margin_median"], e["changed_margin_max"]), flush=True)
        report["models"][model] = entry
        del agent
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
