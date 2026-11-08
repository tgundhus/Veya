"""Measure response enrichment only; excludes inference, JSON encoding and networking."""
import argparse
import json
import platform
import statistics
import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from veya.serve import _add_jev_confidence  # noqa: E402


def result(count):
    answers = {}
    for i in range(count):
        answers[str(i)] = {"type": "choice", "choice": "a", "confidence": 0.3,
                           "probabilities": {"a": 0.8, "b": 0.1, "c": 0.1}}
    return {"model": "test", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}


def measure(fn, repeats, number):
    samples = [elapsed / number * 1e6 for elapsed in timeit.repeat(fn, repeat=repeats, number=number)]
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--number", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.number < 1:
        parser.error("repeats and number must be positive")
    workloads = {}
    for count in (3, 64):
        payload = result(count)
        enriched = _add_jev_confidence(payload)
        assert all(answer["x_jev_confidence"] == 0.7 for answer in enriched["answers"].values())
        assert all("x_jev_confidence" not in answer for answer in payload["answers"].values())
        workloads[str(count)] = {
            "before": measure(lambda: payload, args.repeats, args.number),
            "after": measure(lambda: _add_jev_confidence(payload), args.repeats, args.number),
        }
    report = {"python": platform.python_version(), "platform": platform.system(),
              "repeats": args.repeats, "number": args.number,
              "scope": "response enrichment only; no inference, serialization or networking",
              "workloads": workloads}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
