"""Check the predeclared score pair behind each measured near tie."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from laya.qualification import load_document, validate_corpus


def audit(corpus, report):
    document = validate_corpus(load_document(corpus))
    seal = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if report["harness"]["corpus_file_sha256"] != seal:
        raise ValueError("the report used a different corpus")
    cases = {}
    for case in document["cases"]:
        if "plausible_score_levels" not in case:
            continue
        levels = case["plausible_score_levels"]
        if (len(case["questions"]) != 1 or case["expected"] or len(levels) != 2
                or any(type(level) is not int for level in levels) or levels[0] < 0 or levels[1] != levels[0] + 1):
            raise ValueError(f"invalid unlabelled adjacent pair in {case['id']}")
        qid, question = next(iter(case["questions"].items()))
        if question["type"] != "score" or levels[1] >= len(question["criteria"]):
            raise ValueError(f"invalid score question in {case['id']}")
        cases[case["id"]] = (qid, levels)
    if not cases:
        raise ValueError("no predeclared adjacent pairs")

    threshold = report["limits"]["boundary_margin"]
    profiles = [run for cell in report["cells"].values() for run in cell["runs"]]
    if not profiles or report["coverage"]["skipped_cells"] or report["coverage"]["not_run_cells"]:
        raise ValueError("the qualification report has incomplete profiles")
    rows = {cid: {"expected_levels": levels, "profiles": 0, "matching_pairs": 0, "matched_near_ties": 0,
                  "min_margin": 1., "max_margin": 0.} for cid, (_, levels) in cases.items()}
    for run in profiles:
        for observation in run["observations"]:
            cid = observation["id"]
            if cid not in cases:
                continue
            qid, levels = cases[cid]
            probabilities = observation["answers"][qid]["probabilities"]
            pair = sorted(probabilities, key=probabilities.get, reverse=True)[:2]
            margin = probabilities[pair[0]] - probabilities[pair[1]]
            row = rows[cid]
            correct = sorted(map(int, pair)) == levels
            row["profiles"] += 1
            row["matching_pairs"] += correct
            row["matched_near_ties"] += correct and margin <= threshold
            row["min_margin"] = min(row["min_margin"], margin)
            row["max_margin"] = max(row["max_margin"], margin)
            row.setdefault("returned_pairs", [])
            if pair not in row["returned_pairs"]:
                row["returned_pairs"].append(pair)
    if any(row["profiles"] != len(profiles) for row in rows.values()):
        raise ValueError("the report is missing a boundary observation")
    stable = [cid for cid, row in rows.items() if row["matched_near_ties"] == len(profiles)]
    return {"corpus_sha256": seal, "profiles": len(profiles), "margin_threshold": threshold,
            "stable_matched_cases": stable, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.corpus, load_document(args.report))
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print("Stable near ties with the predeclared score pair:", result["stable_matched_cases"])


if __name__ == "__main__":
    main()
