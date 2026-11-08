"""Compare registration and shortlist helper costs against a Git revision; no model forward.

Run from the repository root:
  python research/scripts/bench_router_shortlist_setup.py --output result.json
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import timeit
import types
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veya import router, shortlist  # noqa: E402


def baseline_module(name, revision):
    source = subprocess.check_output(
        ["git", "show", "%s:laya/%s.py" % (revision, name)], cwd=ROOT
    ).decode("utf-8")
    module = types.ModuleType("veya._bench_baseline_" + name)
    module.__package__ = "laya"
    sys.modules[module.__name__] = module
    exec(compile(source, "<%s:laya/%s.py>" % (revision, name), "exec"), module.__dict__)
    return module


class Dummy:
    def predict(self, state, questions, **kwargs):
        return {"answers": {}}


def cases(router_module, shortlist_module):
    matrix = np.asarray([[1., 0.], [1., 0.], [.9, .1], [.1, .9], [0., 1.]], dtype=np.float32)
    plain = {"q": {"type": "choice", "criteria": ["a", "b", "c", "d"]}}
    ordered = {"q": dict(plain["q"], option_order=[3, 2, 1, 0])}
    embed = lambda texts: matrix
    agent = Dummy()
    return {
        "router_builtin": lambda: router_module.Router(models={"english": "/tmp/x"}),
        "router_distinct_custom": lambda: router_module.Router(models={"papers": "/tmp/x"}),
        "router_typo_custom": lambda: router_module.Router(models={"englsh": "/tmp/x"}),
        "shortlist_plain": lambda: shortlist_module.predict_shortlist(agent, "s", plain, embed, k=2),
        "shortlist_stale_order": lambda: shortlist_module.predict_shortlist(agent, "s", ordered, embed, k=2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="aba90f3")
    parser.add_argument("--number", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.number < 1 or args.repeats < 1:
        parser.error("number and repeats must be positive")
    before = cases(baseline_module("router", args.baseline), baseline_module("shortlist", args.baseline))
    after = cases(router, shortlist)
    results = {}
    # Interleave the versions and alternate their order across repeats to reduce drift.
    # Warning output is ignored; the result measures helper work, not stderr logging.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in before:
            samples = {"before": [], "after": []}
            versions = {"before": before[name], "after": after[name]}
            for fn in versions.values():
                fn()
            for repeat in range(args.repeats):
                order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                for version in order:
                    samples[version].append(timeit.timeit(versions[version], number=args.number) / args.number * 1e6)
            results[name] = {
                version: {"median_us": statistics.median(timings), "samples_us": timings}
                for version, timings in samples.items()
            }
    result = {
        "baseline_ref": args.baseline,
        "baseline_sha": subprocess.check_output(["git", "rev-parse", args.baseline], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "system": platform.system(),
        "calls_per_repeat": args.number,
        "repeats": args.repeats,
        "warning_output": "ignored",
        "scope": "synthetic configuration and helper costs; no model inference or checkpoint loading",
        "caution": "Uncontrolled workstation load. Baseline stale order fails real Agent validation; timing is helper-only.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, timing in results.items():
        print("%s: %.3f -> %.3f us" % (name, timing["before"]["median_us"], timing["after"]["median_us"]))


if __name__ == "__main__":
    main()
