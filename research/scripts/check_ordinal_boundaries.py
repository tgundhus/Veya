"""Check the predeclared score pair behind each measured near tie."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veya.qualification import _limits, _saved_observations, _sha, load_document, validate_corpus


def audit(corpus, report):
    raw = corpus.read_bytes()
    document = validate_corpus(json.loads(raw))
    seal = hashlib.sha256(raw).hexdigest()
    if (report["harness"]["corpus_file_sha256"] != seal or report["corpus_id"] != document["id"]
            or report["corpus_sha256"] != _sha(document)):
        raise ValueError("the report used a different corpus")
    evaluation_ids = {case["id"] for case in document["cases"] if case["split"] == "evaluation"}
    cases = {}
    for case in document["cases"]:
        if "plausible_score_levels" not in case:
            continue
        levels = case["plausible_score_levels"]
        if (case["id"] not in evaluation_ids or len(case["questions"]) != 1 or case["expected"]
                or not isinstance(levels, list) or len(levels) != 2
                or any(type(level) is not int for level in levels) or levels[0] < 0 or levels[1] != levels[0] + 1):
            raise ValueError(f"invalid unlabelled adjacent pair in {case['id']}")
        qid, question = next(iter(case["questions"].items()))
        if question["type"] != "score" or levels[1] >= len(question["criteria"]):
            raise ValueError(f"invalid score question in {case['id']}")
        score_keys = {str(level) for level in range(len(question["criteria"]))}
        cases[case["id"]] = (qid, levels, score_keys)
    if not cases:
        raise ValueError("no predeclared adjacent pairs")

    threshold = report["limits"]["boundary_margin"]
    matrix = report["matrix"]
    requested = {cell["name"] for cell in matrix["cells"]}
    if (not requested or report["matrix_sha256"] != _sha(matrix) or set(report["cells"]) != requested
            or _limits(matrix["limits"]) != report["limits"]
            or set(report["coverage"]["requested_cells"]) != requested or report["coverage"]["skipped_cells"]
            or report["coverage"]["not_run_cells"]):
        raise ValueError("the qualification report has incomplete profiles")
    expected = {(repeat, size, mode, concurrency)
                for repeat in range(matrix["repeats"]) for size in matrix["batch_sizes"]
                for mode in (("ordered",) if size == 1 else ("ordered", "reversed", "neighbour"))
                for concurrency in matrix["concurrencies"]}
    profiles = []
    for cell in report["cells"].values():
        runs = cell.get("runs", [])
        actual = {(run["repeat"], run["batch_size"], run["mode"], run["concurrency"]) for run in runs}
        if not expected or len(runs) != len(expected) or actual != expected:
            raise ValueError("the qualification report has incomplete or duplicate profiles")
        profiles.extend(runs)
    rows = {cid: {"expected_levels": levels, "profiles": 0, "matching_pairs": 0, "matched_near_ties": 0,
                  "min_margin": 1., "max_margin": 0.} for cid, (_, levels, _) in cases.items()}
    for run in profiles:
        observations = _saved_observations(run)
        if {row["id"] for row in observations} != evaluation_ids:
            raise ValueError("a profile is missing corpus observations")
        for observation in observations:
            cid = observation["id"]
            if cid not in cases:
                continue
            qid, levels, score_keys = cases[cid]
            answer = observation["answers"][qid]
            probabilities = answer["probabilities"]
            if answer["answer"]["type"] != "score" or set(probabilities) != score_keys:
                raise ValueError(f"expected probabilities for every score level in {cid}")
            ranked = sorted(probabilities, key=probabilities.get, reverse=True)
            pair = ranked[:2]
            margin = probabilities[pair[0]] - probabilities[pair[1]]
            row = rows[cid]
            unique_pair = len(ranked) == 2 or probabilities[ranked[1]] > probabilities[ranked[2]]
            correct = unique_pair and sorted(map(int, pair)) == levels
            row["profiles"] += 1
            row["matching_pairs"] += correct
            row["matched_near_ties"] += correct and margin <= threshold
            row["min_margin"] = min(row["min_margin"], margin)
            row["max_margin"] = max(row["max_margin"], margin)
            row.setdefault("returned_pairs", [])
            if pair not in row["returned_pairs"]:
                row["returned_pairs"].append(pair)
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
