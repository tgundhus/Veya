"""Compare real English checkpoint answers before and after the Veya import migration.

python research/scripts/check_rename_parity.py --baseline a5477bdc --out parity.json
Uses cached upstream weights offline; the baseline is extracted into a temporary directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
WORKER = """
import importlib, json, sys
sys.path.insert(0, sys.argv[1])
import torch
engine = importlib.import_module(sys.argv[2])
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
cases = json.load(open(sys.argv[3], encoding='utf-8'))
with engine.Router(device='cpu') as router:
    predictions = [router.predict(case['state'], case['questions'], model='english') for case in cases]
    report = {'revision': router.load('english').revision, 'torch': torch.__version__,
              'answers': [p['answers'] for p in predictions], 'usage': [p['usage'] for p in predictions]}
with open(sys.argv[4], 'w', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a5477bdc")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    baseline = subprocess.check_output(["git", "rev-parse", args.baseline], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", baseline, "laya"], cwd=ROOT)
    sys.path.insert(0, str(ROOT))
    from research.scripts.bench_stages import SCENARIOS

    cases = [{"name": name, "state": state, "questions": questions}
             for name, (model, state, questions, _) in SCENARIOS.items()
             if model == "english" and name in (
                 "english ticket, 1 question", "english ticket, 3 questions",
                 "english ticket, 10 questions", "english email, 3 questions")]
    env = {key: value for key, value in os.environ.items() if not key.startswith(("LAYA_", "VEYA_"))}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    with tempfile.TemporaryDirectory(prefix="veya-rename-parity-") as directory:
        temporary = Path(directory)
        with zipfile.ZipFile(io.BytesIO(archive)) as snapshot:
            snapshot.extractall(temporary)
        requests = temporary / "requests.json"
        requests.write_text(json.dumps(cases), encoding="utf-8")
        results = {}
        for label, path, package in (("before", temporary, "laya"), ("after", ROOT, "veya")):
            output = temporary / (label + ".json")
            subprocess.run([sys.executable, "-c", WORKER, str(path), package, str(requests), str(output)],
                           cwd=temporary, env=env, check=True)
            results[label] = json.loads(output.read_text(encoding="utf-8"))
    same = all(results["before"][key] == results["after"][key] for key in ("revision", "answers", "usage"))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "baseline": baseline,
              "device": "cpu", "torch_threads": 2, "cases": cases, "passed": same,
              "answers_compared": sum(len(case["questions"]) for case in cases),
              "candidate_source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"):
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "veya").rglob("*.py"))},
              **results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("%s: %d answers, checkpoint revision and token usage" % (
        "PASS identical" if same else "FAIL changed", report["answers_compared"]))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
