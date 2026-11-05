"""Measure routing-contract preprocessing against the previous reviewed implementation.

No weights, tokenizer, inference or HTTP. Source snapshots and raw alternating timings are recorded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from laya import lang  # noqa: E402
from research.scripts.generate_routing_corpus import NOTE, PROSE  # noqa: E402


def sources():
    names = ("laya/lang.py", "laya/_routing_unicode.py", "laya-ts/src/lang.ts", "laya-ts/src/routing-unicode.ts",
             "research/scripts/bench_routing_contract.py", "research/scripts/bench_routing_contract.mjs",
             "research/scripts/generate_routing_corpus.py", "tests/routing_parity_support.mjs")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def timed(fn, state, iterations):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn(state)
    return (time.perf_counter_ns() - start) / iterations / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="6496fc7a")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--skip-node", action="store_true", help="omit the optional native TypeScript comparison")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    base = subprocess.check_output(["git", "show", args.base + ":laya/lang.py"], cwd=ROOT)
    before = types.ModuleType("routing_before")
    exec(compile(base, "routing_before.py", "exec"), before.__dict__)
    workloads = [
        ("short_string", "Please refund the duplicate charge", 3000),
        ("short_utf8_bytes", "Quero cancelar meu plano agora mesmo".encode(), 3000),
        ("support_ticket", {"note": NOTE * 8, "customer": PROSE["pt"]}, 120),
        ("nested_list", [["field %d" % i for i in range(100)] for _ in range(10)], 15),
        ("wide_first_leaf", {"first": "A" * 10000, **{"tail%d" % i: "x" for i in range(10000)}}, 8),
        ("wide_numeric_keys", {"first": NOTE * 100, **{str(i): "x" for i in range(10000)}}, 8),
        ("multilingual", {"note": NOTE * 100, "customer": PROSE["hi"]}, 50),
        ("line_after_emoji", {"note": NOTE * 100, "customer": "😀" * 4001 + "\n" + PROSE["pt"]}, 50),
    ]
    initial = sources()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "base_revision": args.base,
              "base_source_sha256": hashlib.sha256(base).hexdigest(), "source_sha256": initial,
              "contract": lang.ROUTING_CONTRACT_VERSION, "repeats": args.repeats,
              "limits": "Dependency-free preprocessing only, no inference-speed claim. "
                        "Nine alternating samples by default. Per-sample mean call time, then median across samples. "
                        "Uncontrolled workstation load, no memory measurements. Canonical numeric-key ordering "
                        "must inspect mapping keys; values remain lazy. Wide-key workloads intentionally expose that cost.",
              "cases": []}
    for name, state, iterations in workloads:
        row = {"name": name, "iterations": iterations, "operations": {}}
        for operation in ("state_text", "analyse"):
            old_fn, new_fn = getattr(before, operation), getattr(lang, operation)
            prior, current = old_fn(state), new_fn(state)
            measurements = {"before": [], "after": []}
            for _ in range(3):
                old_fn(state)
                new_fn(state)
            for repeat in range(args.repeats):
                for label, fn in (("before", old_fn), ("after", new_fn)) if repeat % 2 == 0 else (
                        ("after", new_fn), ("before", old_fn)):
                    measurements[label].append(timed(fn, state, iterations))
            medians = {label: statistics.median(values) for label, values in measurements.items()}
            row["operations"][operation] = {"mean_call_samples_us": measurements, "median_us": medians,
                                            "after_over_before": medians["after"] / medians["before"],
                                            "outputs_equal": prior == current}
        result["cases"].append(row)
        print(name, json.dumps(row["operations"]), flush=True)
    node = shutil.which("node")
    compiler = ROOT / "laya-ts/node_modules/typescript/bin/tsc"
    if not args.skip_node and node and compiler.is_file():
        subprocess.run([node, str(compiler), "-p", "tsconfig.json"], cwd=ROOT / "laya-ts", check=True,
                       capture_output=True, encoding="utf-8", timeout=90)
        source = subprocess.check_output(["git", "show", args.base + ":laya-ts/src/lang.ts"], cwd=ROOT)

        def serializable(value):
            if isinstance(value, bytes):
                return {"$bytes": list(value)}
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, list):
                return [serializable(item) for item in value]
            return value

        inputs = {"source": source.decode("utf-8"), "repeats": args.repeats,
                  "workloads": [{"name": name, "state": serializable(state), "iterations": iterations}
                                for name, state, iterations in workloads]}
        completed = subprocess.run([node, str(ROOT / "research/scripts/bench_routing_contract.mjs")], cwd=ROOT,
                                   input=json.dumps(inputs, ensure_ascii=True), capture_output=True,
                                   encoding="utf-8", timeout=240)
        if completed.returncode:
            raise RuntimeError("TypeScript benchmark failed: " + completed.stderr[-4000:])
        result["typescript"] = json.loads(completed.stdout)
        result["typescript"]["base_source_sha256"] = hashlib.sha256(source).hexdigest()
    else:
        result["typescript"] = None
    assert initial == sources(), "source changed during timing; rerun after freeze"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
