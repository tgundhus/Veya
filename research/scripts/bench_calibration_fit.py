"""Compare calibration fits with a Git baseline on seeded CPU records."""
import argparse
from decimal import Decimal
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from veya import calibrate as candidate  # noqa: E402
from veya.common import QTYPES  # noqa: E402

BASELINE_REF = "2dd6c79da2cc73dbc88785da50201bdf268f31c6"


def baseline_fitter(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:veya/calibrate.py"], cwd=ROOT)
    module = ModuleType("veya._calibration_source_baseline")
    exec(compile(source, "<baseline-calibrate>", "exec"), vars(module))
    return module, hashlib.sha256(source).hexdigest()


def records(n):
    rng = np.random.default_rng(241125)
    types = ((QTYPES["choice"], (3, 5)), (QTYPES["noul"], (2,)), (QTYPES["score"], (6, 8, 10)))
    result = []
    for i in range(n):
        qtype, widths = types[i % len(types)]
        k = widths[(i // len(types)) % len(widths)]
        z = rng.normal(0, 1.7, size=k).astype(np.float32)
        p = np.exp(z / 1.3 - (z / 1.3).max(), dtype=np.float64)
        y = rng.choice(k, p=p / p.sum())
        t = np.zeros(k, np.float32)
        t[y] = 1.0
        if i % 11 == 0:
            t[y] = 0.8
            t[(y + 1) % k] = 0.2
        if i % 7 == 0:
            z = np.pad(z, (0, 2), constant_values=-99)
            t = np.pad(t, (0, 2))
        result.append((qtype, z, t, k))
    return result


def error(call):
    try:
        return call()
    except (ValueError, TypeError, OverflowError) as exc:
        return (type(exc).__name__, str(exc))


def validation_parity(baseline):
    z, t = np.array([1, 2], np.float32), np.array([1, 0], np.float32)
    # Empty, dtype/shape, full-vector finiteness, invalid k and near-tolerance sums.
    bad_records = [
        (0, z, t, None), (0, z, t, 0), (0, z, t, True), (0, z, t, 3),
        (True, z, t), (3, z, t), (0, [], []), (0, [True, True], t),
        (0, [[1.0, 2.0]], t), (0, [10**400, 1], t), (0, z, t[:1]),
        (0, [1.0, 2.0, float("inf")], [1.0, 0.0, 0.0], 2),
        (0, [1.0, 2.0, 0.0], [1.0, 0.0, float("nan")], 2),
        (0, z, [1.25, -0.25]), (0, z, [1.0, 2e-5]),
    ]
    for rec in bad_records:
        got = error(lambda: candidate.fit_temperature_map([rec]))
        want = error(lambda: baseline.fit_temperature_map([rec]))
        assert got == want and got[0] == "ValueError", (rec, got, want)

    for step in range(-200, 201):
        s = 1.0 + step * 2**-24
        rec = (0, z, [s, 0.0])
        assert error(lambda: candidate.fit_temperature_map([rec])) == error(
            lambda: baseline.fit_temperature_map([rec])), s

    ties = [(0.94, 1), (0.91, 0), (0.91, 1), (0.88, 1), (0.88, 1), (0.73, 0)]
    rng = np.random.default_rng(7)
    cases = [[], ties, [(0.88, 1)] * 20, [(0.99, 0)] * 20]
    cases.extend([(round(float(c), 2), int(y)) for c, y in zip(rng.random(100), rng.integers(2, size=100))]
                 for _ in range(20))
    for pairs in cases:
        for risk in (0.0, 0.1, 0.2, 0.3333333333333333, 0.5, 1.0):
            for conservative in (True, False):
                a = candidate._select_abstention_threshold(pairs, risk, conservative)
                b = baseline._select_abstention_threshold(pairs, risk, conservative)
                assert a == b, (pairs, risk, conservative, a, b)

    precision_tie = [(0.8, 1)] * 24 + [(0.8, 0)]
    for risk in (np.float16(0.04), np.float32(0.04), np.float64(0.04),
                 Decimal("0.04"), Decimal.from_float(0.04)):
        got = candidate._select_abstention_threshold(precision_tie, risk, False)
        want = baseline._select_abstention_threshold(precision_tie, risk, False)
        assert got == want, (type(risk), got, want)

    broken_map = {"choice:2": {"bins": 1, "values": [float("nan")]}}
    bad = error(lambda: candidate.fit_abstention_thresholds(
        [(0, [2., 1.], [1., 0.])] * 100, [1., 1., 1.], {}, binning_map=broken_map))
    assert bad[0] == "ValueError", bad

    bin_edges = [(0, [1.0], [1.0]), (0, [0.0, 0.0], [1.0, 0.0]), (0, [0.0, 0.0], [0.0, 1.0])]
    for bins in (True, 1, 2, 15, np.int32(4)):
        got = candidate.fit_binning_map(bin_edges, [1., 1., 1.], {}, bins=bins, min_bucket_n=1)
        want = baseline.fit_binning_map(bin_edges, [1., 1., 1.], {}, bins=bins, min_bucket_n=1)
        assert got == want, (bins, got, want)

    def reused_buffer():
        for i in range(40):
            z[:] = [i / 15, -i / 15]
            t[:] = [i % 2, 1 - i % 2]
            yield QTYPES["noul"], z, t, 2

    assert candidate.fit_temperature_map(reused_buffer()) == baseline.fit_temperature_map(reused_buffer())


def canonical(value):
    # JSON parser markers make matching no-holdout NaN metrics comparable.
    return json.loads(json.dumps(value), parse_constant=lambda name: name)


def paired(before, after, repeats):
    implementations = {"before": before, "after": after}
    expected = canonical(before())
    actual = canonical(after())
    assert expected == actual, (expected, actual)
    samples = {name: [] for name in implementations}
    for i in range(repeats):
        order = implementations if not i % 2 else reversed(implementations)
        for name in order:
            started = perf_counter()
            result = implementations[name]()
            samples[name].append((perf_counter() - started) * 1000)
            assert canonical(result) == expected, (name, result, expected)
    return {"samples_ms": samples, "median_ms": {k: median(v) for k, v in samples.items()},
            "faster_pairs": sum(a < b for a, b in zip(samples["after"], samples["before"])),
            "exact_outputs": True, "output": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    baseline, baseline_hash = baseline_fitter(args.baseline_ref)
    validation_parity(baseline)
    runs = {}
    for n in (1200, 12000):
        recs = records(n)
        temps = baseline.fit_temperature_map(recs, compute_ece=True, seed=7)
        scalars, by_options = temps["temperature"], temps["temperature_by_options"]
        binning = baseline.fit_binning_map(recs, scalars, by_options, bins=15)
        named = {
            "temperature_holdout": lambda m: m.fit_temperature_map(recs, compute_ece=True, seed=7),
            "binning": lambda m: m.fit_binning_map(recs, scalars, by_options, bins=15),
            "abstention_raw": lambda m: m.fit_abstention_thresholds(recs, scalars, by_options, target_error=0.2),
            "abstention_binned": lambda m: m.fit_abstention_thresholds(
                recs, scalars, by_options, target_error=0.2, binning_map=binning),
        }
        runs[n] = {}
        for name, call in named.items():
            measurement = paired(lambda: call(baseline), lambda: call(candidate), args.repeats)
            medians = measurement["median_ms"]
            print(f"{n} rows {name}: {medians['before']:.3f} -> {medians['after']:.3f} ms "
                  f"({measurement['faster_pairs']}/{args.repeats} faster pairs)", flush=True)
            runs[n][name] = measurement
    payload = {
        "baseline_ref": subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        "baseline_calibrate_sha256": baseline_hash,
        "after_calibrate_sha256": hashlib.sha256(
            (ROOT / "veya/calibrate.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "torch": torch.__version__, "torch_threads": args.threads},
        "protocol": {"records_seed": 241125, "split_seed": 7, "alternating_pairs": args.repeats,
                     "bins": 15, "target_error": 0.2,
                     "validation": "malformed refusals, float32 sum boundary, confidence ties, "
                                   "NumPy/Decimal risk boundaries, invalid binned NaN refusal, "
                                   "bin edges and reused buffers"},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
