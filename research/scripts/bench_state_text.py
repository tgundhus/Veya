"""Measure bounded text flattening against the former eager implementation, without models.

Run: python research/scripts/bench_state_text.py --out research/results/state_text.json
These timings measure state_text only, not language detection or model inference.
"""
import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import timeit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from laya.lang import state_text  # noqa: E402


def eager_text_leaves(state, depth=0, paths=None, path=()):
    """Frozen pre-optimization reference, including diagnostic source paths."""
    if depth > 6 or state is None:
        return []
    if isinstance(state, str):
        if paths is not None:
            paths.append(path)
        return [state]
    if isinstance(state, (bytes, bytearray)):
        try:
            text = bytes(state).decode("utf-8")
        except UnicodeDecodeError:
            return []
        if paths is not None:
            paths.append(path)
        return [text]
    result = []
    if isinstance(state, Mapping):
        if paths is None:
            for value in state.values():
                result.extend(eager_text_leaves(value, depth + 1))
        else:
            for key, value in state.items():
                bounded = ((type(key) is str and len(key) <= 128)
                           or (type(key) is int and key.bit_length() <= 63))
                child = path + (key,) if path is not None and bounded else None
                result.extend(eager_text_leaves(value, depth + 1, paths, child))
    elif isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            child = path + (index,) if paths is not None and path is not None else path
            result.extend(eager_text_leaves(value, depth + 1, paths, child))
    return result


def eager_state_text(state, max_chars=4000):
    """Frozen pre-optimization reference: flatten everything before applying the budget."""
    parts = []
    budget = max_chars
    for leaf in eager_text_leaves(state):
        if budget <= 0:
            break
        if len(leaf) > budget:
            parts.append(leaf[:budget])
            break
        parts.append(leaf)
        budget -= len(leaf) + 1
    return " ".join(parts)[:max_chars]


def cases():
    return {
        "short_string": "Please route this duplicate payment refund to billing.",
        "short_bytes": b"Please route this duplicate payment refund to billing.",
        "nested_10000": [{"text": "request " + str(i)} for i in range(10000)],
        "long_first_leaf": ["A" * 8000] + [{"text": "request " + str(i)} for i in range(10000)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = []
    for name, state in cases().items():
        budgets = [-1, 0, 1, 5, 4000, 10000]
        for budget in budgets:
            assert state_text(state, budget) == eager_state_text(state, budget), (name, budget)
        number = 10000 if name.startswith("short") else 100
        samples = {"before": [], "after": []}
        # Alternate order so one implementation does not always get the warmer CPU.
        for repeat in range(args.repeats):
            order = (("before", eager_state_text), ("after", state_text))
            for label, fn in (order if repeat % 2 == 0 else reversed(order)):
                samples[label].append(timeit.timeit(lambda: fn(state), number=number) / number)
        before, after = (statistics.median(samples[label]) for label in ("before", "after"))
        rows.append({"case": name, "iterations_per_sample": number, "budget_parity": budgets,
                     "before_median_us": before * 1e6, "after_median_us": after * 1e6,
                     "speedup": before / after, "samples_seconds_per_call": samples})
        print(name, "%.3f -> %.3f us (%.2fx)" % (before * 1e6, after * 1e6, before / after))
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "repeats": args.repeats, "rows": rows,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT / "laya/lang.py", Path(__file__))},
        "upstream_inspiration": "https://github.com/NandhaKishorM/laya/commit/9580d43fafe16645ee1a03063d0c479c8f8532d1",
        "limits": "state_text preprocessing only; no model, inference throughput or accuracy measurement. "
                  "Workstation load and filesystem caches were uncontrolled.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
