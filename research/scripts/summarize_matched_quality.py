"""Audit the shared Jev/Laya/Veya protocol and write reproducible quality comparisons."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/benchmarks/feishu_zh"
sys.path.insert(0, str(PROTOCOL))
from audit import audit_run, verify_archive  # noqa: E402


def summarize(veya_run):
    verify_archive()
    folders = {"jev": PROTOCOL / "results/v1/jev", "laya": PROTOCOL / "results/v1/laya",
               "veya": veya_run}
    sources, rows, metadata, models, requests = {}, {}, {}, {}, {}
    for backend, folder in folders.items():
        models[backend] = audit_run(folder)
        for name in ("metadata.json", "raw.jsonl"):
            path = folder / name
            key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
            sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata[backend] = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        assert metadata[backend]["finished_at"] and metadata[backend]["repeats"] == 3
        rows[backend] = [json.loads(line) for line in (folder / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows[backend]) == 384 and all(row["status"] == "ok" for row in rows[backend])
        requests[backend] = {(row["id"], row["mode"], row["repeat"]): row["request_sha256"] for row in rows[backend]}
        groups = defaultdict(list)
        for row in rows[backend]:
            groups[row["id"], row["mode"]].append(row["response"]["answers"])
        models[backend]["exact_repeat_groups"] = {
            "identical": sum(all(answer == answers[0] for answer in answers) for answers in groups.values()),
            "total": len(groups)}
    assert requests["jev"] == requests["laya"] == requests["veya"]
    environment = PROTOCOL / "results/v1/environment_check.json"
    expected = json.loads(environment.read_text(encoding="utf-8"))
    assert metadata["veya"]["checkpoint_model_sha256"] == expected["weight_sha256"]
    assert metadata["veya"]["checkpoint_revision"] == metadata["laya"]["checkpoint_revision"]
    sources[environment.relative_to(ROOT).as_posix()] = hashlib.sha256(environment.read_bytes()).hexdigest()
    for name in ("research/scripts/summarize_matched_quality.py", "research/benchmarks/feishu_zh/audit.py",
                 "research/benchmarks/feishu_zh/metrics.py", "research/benchmarks/feishu_zh/prompts.py"):
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    changes = {}
    for mode in ("choice", "four_noul"):
        old = {row["id"]: row["predicted"] for row in rows["laya"] if row["mode"] == mode and row["repeat"] == 0}
        new = {row["id"]: row["predicted"] for row in rows["veya"] if row["mode"] == mode and row["repeat"] == 0}
        assert len(old) == len(new) == 64
        changes[mode] = sum(old[key] != new[key] for key in old)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "models": models,
            "source_sha256": sources, "same_request_hashes": True,
            "same_laya_veya_weights_sha256": expected["weight_sha256"],
            "paired_label_changes_laya_vs_veya": changes,
            "method": "64 frozen synthetic Chinese cases, two modes, three repeats, fixed first-repeat quality. "
                      "Recorded Jev/Laya calls from September 2026; Veya measured November 2026. "
                      "Same prompts and scoring, pinned identical Laya/Veya weights; different hardware/dates. "
                      "No cross-engine speed ranking or general multilingual accuracy claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veya-run", type=Path, default=ROOT / "research/results/matched_feishu_veya_cpu_20261107")
    parser.add_argument("--out", type=Path, default=ROOT / "research/results/matched_quality_comparison_20261107.json")
    args = parser.parse_args()
    report = summarize(args.veya_run.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for backend, model in report["models"].items():
        print(backend, {mode: (model[mode]["correct"], model[mode]["n"]) for mode in ("choice", "four_noul")},
              model["exact_repeat_groups"])
