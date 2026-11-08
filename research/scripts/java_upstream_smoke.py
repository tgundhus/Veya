"""Compare the local Java SDK with a supplied English PyTorch checkpoint.

Build veya-java first. Pass --classpath containing its runtime JAR/classes and the ONNX Runtime
JAR, --jdk pointing to a JDK, --checkpoint and --graph pointing to local model directories,
--checkpoint-revision identifying the checkpoint, and --output for the sanitized JSON result.
This checks returned answers and usage, plus a small warmed batch timing comparison.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "graph", "jdk", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--classpath", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.graph.is_dir():
        parser.error("checkpoint and graph must be local directories")

    import torch
    from veya import Agent
    from veya.shortlist import predict_tournament

    torch.set_num_threads(2)
    states = [
        {"message": "I was charged twice for my monthly subscription. Please refund the duplicate payment."},
        {"message": "The app crashes every time I sign in. I need technical help."},
    ]
    questions = {
        "team": {"type": "choice", "instructions": "Which team should handle `message`?",
                 "criteria": {"billing": "Payments, invoices, or refunds",
                              "technical": "Software bugs and access problems", "other": "Anything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is the support request in `message`?",
                    "criteria": ["Low: routine request", "Medium: customer needs help soon",
                                 "High: service is unusable"]},
        "refund": {"type": "noul", "instructions": "Is the customer requesting a refund in `message`?"},
    }
    one = {
        "topic": {"type": "choice", "instructions": "What is this about?", "criteria": {"billing": "a billing issue"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["urgent"]},
    }
    labels = ["billing", "refunds", "fraud", "cards", "transfers", "loans", "savings", "kyc", "support", "legal",
              "sales", "tech", "identity", "fees", "cash", "statements", "login", "cancel", "lost", "other"]
    wide = {"intent": {"type": "choice", "instructions": "What does the customer need?",
                       "criteria": {label: label for label in labels}}}
    agent = Agent(str(args.checkpoint), device="cpu", compile=False)

    def output(prediction):
        return {"answers": prediction["answers"], "usage": prediction["usage"]}

    expected = {
        "singles": [output(agent.predict(state, questions)) for state in states],
        "batch": [output(p) for p in agent.predict_batch(states, questions, batch_size=2)],
        "one_option": [output(agent.predict(state, one)) for state in states],
    }
    expected["router_singles"] = expected["singles"]
    expected["router_batch"] = expected["batch"]
    tournament = predict_tournament(agent, states[0], wide, group_size=8)
    bracket = tournament["tournament"]["intent"]
    expected["tournament"] = {"prediction": output(tournament), "labels": bracket["labels"],
                              "total": bracket["n"], "rounds": bracket["rounds"]}
    # JSON object keys are strings in Java's output, including score legends and probabilities.
    expected = json.loads(json.dumps(expected))

    suffix = ".exe" if os.name == "nt" else ""
    java = args.jdk / "bin" / ("java" + suffix)
    javac = args.jdk / "bin" / ("javac" + suffix)
    source = Path(__file__).with_name("JavaUpstreamSmoke.java")
    with tempfile.TemporaryDirectory(prefix="veya-java-smoke-") as directory:
        work = Path(directory)
        reference = work / "reference.json"
        target = work / "native.json"
        reference.write_text(json.dumps({"states": states, "questions": questions}), encoding="utf-8")
        subprocess.run([str(javac), "-Xlint:all", "-Werror", "-cp", args.classpath, "-d", directory, str(source)],
                       check=True)
        subprocess.run([str(java), "-cp", directory + os.pathsep + args.classpath, "JavaUpstreamSmoke",
                        str(args.checkpoint), str(args.graph), str(reference), str(target)], check=True)
        native = json.loads(target.read_text(encoding="utf-8"))

    differences = []
    largest = 0.0

    def compare(got, want, field):
        nonlocal largest
        if isinstance(want, dict):
            if (not isinstance(got, dict) or set(got) != set(want)
                    or field.endswith((".answers", ".probabilities", ".legend")) and list(got) != list(want)):
                differences.append({"field": field, "error": "object keys or order differ"})
                return
            for key, value in want.items():
                compare(got[key], value, field + "." + key)
        elif isinstance(want, list):
            if not isinstance(got, list) or len(got) != len(want):
                differences.append({"field": field, "error": "list length differs"})
                return
            for index, (actual, reference) in enumerate(zip(got, want)):
                compare(actual, reference, field + "." + str(index))
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            if not isinstance(got, (int, float)) or isinstance(got, bool) or not math.isfinite(got):
                differences.append({"field": field, "actual": got, "expected": want})
                return
            delta = abs(got - want)
            largest = max(largest, delta)
            tolerance = 0 if ".usage." in field or field.endswith((".total", ".rounds")) else 1e-4
            if delta > tolerance:
                differences.append({"field": field, "actual": got, "expected": want, "difference": delta})
        elif got != want:
            differences.append({"field": field, "actual": got, "expected": want})

    for mode, reference in expected.items():
        compare(native[mode], reference, mode)
    benchmark = native["benchmark"]
    sequential = statistics.median(benchmark["sequential_whole_call_ms"])
    batched = statistics.median(benchmark["batch_whole_call_ms"])
    benchmark["median_sequential_whole_call_ms"] = sequential
    benchmark["median_batch_whole_call_ms"] = batched
    benchmark["median_sequential_per_request_ms"] = sequential / len(states)
    benchmark["median_batch_per_request_ms"] = batched / len(states)
    benchmark["sequential_to_batch_ratio"] = sequential / batched
    report = {
        "schema": 1, "platform": platform.system(), "checkpoint": "convaiinnovations/laya (English)",
        "checkpoint_revision": args.checkpoint_revision,
        "upstream_revision": "68804629e8ccd9d616d48a40e87de9fabbeae069",
        "graph": native["graph"], "states": states, "questions": questions,
        "one_option_questions": one, "tournament_questions": wide, "tournament_group_size": 8,
        "reference_backend": "PyTorch CPU, 2 threads, compile=False", "sdk": "Java, ONNX Runtime 1.20.0",
        "validation": {"tolerance": 1e-4, "usage_tolerance": 0, "max_difference": largest,
                       "errors": differences, "passed": not differences},
        "limits": ["Two states and one tournament; no accuracy measurement.",
                   "Returned probabilities are rounded; raw-logit equality is not asserted.",
                   "Multilingual and typed-decisions checkpoint parity is not covered.",
                   "Three warm timing repeats on an active workstation are not a throughput or deployment benchmark."],
        "hardware": {"processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "benchmark": benchmark,
        "expected": expected, "observed": {mode: native[mode] for mode in expected},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # One JSON row per result section keeps repeated answer tables compact and reproducible.
    rows = ["  " + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            for key, value in report.items()]
    args.output.write_text("{\n" + ",\n".join(rows) + "\n}\n", encoding="utf-8")
    print(json.dumps(report["validation"]))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
