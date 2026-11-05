"""Run the release qualification gate on existing English CPU artifacts only.

Example: python research/scripts/qualify_local.py --checkpoint /models/laya \
  --onnx /models/english.onnx --out research/results/qualification.json
No checkpoint downloads, export, training or hosted service is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from laya.decision_policy import DECISION_POLICIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "research/evals/qualification-v1.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--decision-policy", choices=DECISION_POLICIES, default=None,
                        help="explicit local decision policy; omitted uses the checkpoint's default")
    parser.add_argument("--diagnostic", action="store_true",
                        help="run agreement diagnostics without claiming fixture quality qualification")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_THREADS=str(args.threads))
    import torch
    from laya import qualification
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = str(args.checkpoint.resolve())
    cells = [{"name": "eager-cpu-fp32", "checkpoint": checkpoint, "backend": "eager",
              "device": "cpu", "precision": "fp32", "required": True}]
    if args.onnx:
        cells.append({"name": "onnx-cpu-graph", "checkpoint": checkpoint, "backend": "onnx",
                      "device": "cpu", "precision": "graph", "onnx_path": str(args.onnx.resolve()), "required": True})
    if args.decision_policy is not None:
        for cell in cells:
            cell["decision_policy"] = args.decision_policy
    limits = {"max_label_flip_rate": 0, "max_probability_drift": 0.001, "max_score_drift": 0.001,
              "max_accuracy_drop": 0, "boundary_margin": 0.05, "min_boundary_cases": 0 if args.diagnostic else 1}
    if not args.diagnostic:
        limits["min_accuracy"] = 0.8
    matrix = {"schema": qualification.MATRIX_SCHEMA, "reference": "eager-cpu-fp32", "cells": cells,
              "batch_sizes": args.batch_sizes, "concurrencies": args.concurrencies,
              "repeats": args.repeats, "limits": limits}
    corpus = qualification.load_document(args.corpus)
    report = qualification.run_matrix(corpus, matrix)
    report["harness"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "corpus_file_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                         "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                         "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
                         "limits": "Synthetic rubric fixture, small local sample; no general accuracy or HTTP p99 claim. "
                                   "Workstation load is uncontrolled. Zero cache use, fp32/eager versus the pinned graph."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, cell in report["cells"].items():
        print(name, cell["status"], cell.get("error", ""), flush=True)
        if cell.get("runs"):
            drift = max(run["comparison"]["max_probability_drift"] for run in cell["runs"])
            print("  first metrics", cell["runs"][0]["metrics"], "max probability drift", drift, flush=True)
    print("Measured boundary cases:", report["coverage"].get("boundary_cases"), flush=True)
    print("Passed:", report["passed"], "Production qualified:", report["production_qualified"], flush=True)
    print("Saved", args.out, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
