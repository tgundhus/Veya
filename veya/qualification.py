"""Local release qualification over fixed, family-disjoint decision fixtures.

Importing this module does not load a tensor runtime. Matrix cells are constructed
one at a time, from local artifacts, so an unavailable required target cannot be
silently replaced by another backend or counted as a passing target.
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time

from . import evals
from .decision_policy import DECISION_POLICIES, resolve_decision_policy
from .evals import EvalError

CORPUS_SCHEMA = "veya-qualification-corpus/1"
MATRIX_SCHEMA = "veya-qualification-matrix/1"
REPORT_SCHEMA = "veya-qualification-report/1"
_SPLITS = {"training", "calibration", "evaluation"}
_LIMITS = {"max_label_flip_rate": 0.0, "max_probability_drift": 0.001,
           "max_score_drift": 0.001, "min_accuracy": None, "max_accuracy_drop": 0.0,
           "max_p95_ms": None, "max_p99_ms": None, "min_boundary_cases": 0,
           "boundary_margin": 0.05}
_DRIFT_LIMITS = {"label_flip_rate": "max_label_flip_rate", "max_probability_drift": "max_probability_drift",
                 "max_score_drift": "max_score_drift", "max_confidence_drift": "max_probability_drift",
                 "max_action_drift": "max_probability_drift", "operational_flag_flip_rate": "max_label_flip_rate"}


def _encoded(value):
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise EvalError("qualification data must be finite, UTF-8 JSON: %s" % exc) from exc


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def load_document(path):
    """Read a JSON document, refusing duplicate members and non-finite numbers."""
    def members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON member %r" % key)
            result[key] = value
        return result
    try:
        with open(path, encoding="utf-8-sig") as handle:
            result = json.load(handle, object_pairs_hook=members,
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise EvalError("cannot read qualification document %r: %s" % (str(path), exc)) from exc
    _encoded(result)
    return result


def _name(value, where):
    if not isinstance(value, str) or not value.strip():
        raise EvalError("%s must be a nonempty string" % where)
    return value


def _integer(value, where, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise EvalError("%s must be an integer >= %d%s" %
                        (where, minimum, " and <= %d" % maximum if maximum is not None else ""))
    return value


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def validate_corpus(document):
    """Validate cases, gold types and split leakage; return an independent copy.

    Families are supplied provenance, rather than inferred semantic equivalence.
    Every variant of a source family must have the same split. Byte-equivalent
    state/question inputs may not appear under different splits either.
    """
    from .mcp.tools import ToolError, validate_questions

    if not isinstance(document, dict) or document.get("schema") != CORPUS_SCHEMA:
        raise EvalError("expected corpus schema %r" % CORPUS_SCHEMA)
    _encoded(document)
    _name(document.get("id"), "corpus.id")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvalError("corpus.cases must be a nonempty list")
    ids, families, inputs = set(), {}, {}
    for index, case in enumerate(cases):
        where = "corpus.cases[%d]" % index
        if not isinstance(case, dict):
            raise EvalError("%s must be an object" % where)
        cid = _name(case.get("id"), where + ".id")
        family = _name(case.get("family"), where + ".family")
        _name(case.get("variant"), where + ".variant")
        split = case.get("split")
        if cid in ids:
            raise EvalError("duplicate case id %r" % cid)
        ids.add(cid)
        if split not in _SPLITS:
            raise EvalError("%s.split must be training, calibration or evaluation" % where)
        if family in families and families[family] != split:
            raise EvalError("family %r leaks across splits" % family)
        families[family] = split
        if not isinstance(case.get("state"), (str, dict, list)):
            raise EvalError("%s.state must be a JSON string, object or list" % where)
        try:
            validate_questions(case.get("questions"))
        except ToolError as exc:
            raise EvalError("%s: %s" % (where, exc)) from exc
        example = evals.Example.from_dict(case, where)
        if example.model is not None:
            raise EvalError("qualification cases cannot override a matrix cell's checkpoint")
        for qid, expected in example.expected.items():
            question = example.questions[qid]
            kind = question["type"]
            if kind == "choice":
                valid = isinstance(expected, str) and expected in question["criteria"]
            elif kind == "noul":
                valid = type(expected) is bool
            else:
                valid = (_finite(expected)
                         and 0 <= expected <= len(question["criteria"]) - 1)
            if not valid:
                raise EvalError("%s.expected[%r] is not a valid %s label" % (where, qid, kind))
        signature = _sha([case["state"], case["questions"]])
        if signature in inputs and inputs[signature] != split:
            raise EvalError("identical input leaks across splits at case %r" % cid)
        inputs[signature] = split
    if not any(case["split"] == "evaluation" for case in cases):
        raise EvalError("corpus needs at least one evaluation case")
    return deepcopy(document)


def split_families(document, *, evaluation_fraction=0.2, calibration_fraction=0.2, seed=0):
    """Assign whole families by a seeded metadata-only hash, never by predictions."""
    document = validate_corpus(document)
    _integer(seed, "seed", minimum=0)
    for name, value in (("evaluation_fraction", evaluation_fraction),
                        ("calibration_fraction", calibration_fraction)):
        if not _finite(value) or not 0 <= value < 1:
            raise EvalError("%s must be finite in [0, 1)" % name)
    if evaluation_fraction <= 0 or evaluation_fraction + calibration_fraction >= 1:
        raise EvalError("split fractions must leave both evaluation and training families")
    families = sorted({case["family"] for case in document["cases"]},
                      key=lambda family: _sha([seed, family]))
    n_eval = max(1, int(len(families) * evaluation_fraction))
    n_cal = max(1, int(len(families) * calibration_fraction)) if calibration_fraction else 0
    if n_eval + n_cal >= len(families):
        raise EvalError("not enough whole families for the requested splits")
    assignment = {family: ("evaluation" if i < n_eval else
                           "calibration" if i < n_eval + n_cal else "training")
                  for i, family in enumerate(families)}
    for case in document["cases"]:
        case["split"] = assignment[case["family"]]
    document["split_policy"] = {"seed": seed, "evaluation_fraction": evaluation_fraction,
                                "calibration_fraction": calibration_fraction,
                                "method": "whole-family seeded hash; metadata only"}
    return validate_corpus(document)


def _limits(limits):
    if limits is not None and not isinstance(limits, dict):
        raise EvalError("limits must be an object")
    unknown = set(limits or {}) - set(_LIMITS)
    if unknown:
        raise EvalError("unknown qualification limit(s): %s" % ", ".join(sorted(unknown)))
    values = dict(_LIMITS, **(limits or {}))
    for name, value in values.items():
        if value is None and name in ("min_accuracy", "max_p95_ms", "max_p99_ms"):
            continue
        if not _finite(value) or value < 0:
            raise EvalError("%s must be finite and nonnegative" % name)
        if name in ("max_label_flip_rate", "max_probability_drift", "min_accuracy", "max_accuracy_drop", "boundary_margin") and value > 1:
            raise EvalError("%s must be <= 1" % name)
    _integer(values["min_boundary_cases"], "min_boundary_cases", minimum=0)
    return values


def _number(value, where, lower=None, upper=None):
    if not _finite(value):
        raise EvalError("%s must be a finite number" % where)
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise EvalError("%s is outside its valid range" % where)
    return float(value)


def _answers(result, questions):
    _encoded(result)
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise EvalError("runner must return exactly the requested answer ids")
    clean = {}
    for qid, question in questions.items():
        answer, kind = answers[qid], question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise EvalError("runner returned the wrong answer type for %r" % qid)
        if kind == "noul":
            p = _number(answer.get("noul"), qid + ".noul", 0, 1)
            vector, label, score = {"false": 1 - p, "true": p}, p >= 0.5, None
        else:
            labels = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
                raise EvalError("runner returned incomplete probabilities for %r" % qid)
            vector = {label: _number(probabilities[label], qid + ".probabilities", 0, 1) for label in labels}
            if abs(sum(vector.values()) - 1.0) > max(0.001, len(labels) * 0.000051):
                raise EvalError("runner returned probabilities that do not sum to one for %r" % qid)
            if kind == "choice":
                label, score = answer.get("choice"), None
                if label not in vector:
                    raise EvalError("runner returned an unknown choice for %r" % qid)
            else:
                label = None
                score = _number(answer.get("score"), qid + ".score", 0, len(labels) - 1)
        ordered = sorted(vector.values(), reverse=True)
        operational = {}
        for field in ("confidence", "answer_confidence"):
            if field in answer:
                operational[field] = _number(answer[field], qid + "." + field, 0, 1)
        if "action" in answer:
            if not isinstance(answer["action"], dict) or "act_probability" not in answer["action"]:
                raise EvalError("runner returned an invalid action probability for %r" % qid)
            operational["act_probability"] = _number(answer["action"]["act_probability"], qid + ".act_probability", 0, 1)
        flags = {}
        if "low_confidence" in answer:
            if type(answer["low_confidence"]) is not bool:
                raise EvalError("low_confidence must be a boolean")
            flags["low_confidence"] = answer["low_confidence"]
        if "abstention" in answer:
            if not isinstance(answer["abstention"], str):
                raise EvalError("abstention must be a string")
            flags["abstention"] = answer["abstention"]
        if "confidence_calibrated" in answer:
            if type(answer["confidence_calibrated"]) is not bool:
                raise EvalError("confidence_calibrated must be a boolean")
            flags["confidence_calibrated"] = answer["confidence_calibrated"]
        clean[qid] = {"answer": deepcopy(answer), "probabilities": vector,
                      "label": label, "score": score,
                      "operational": operational, "flags": flags,
                      "margin": ordered[0] - ordered[1] if len(ordered) > 1 else 1.0}
    return clean


def _percentiles(values):
    ordered = sorted(values)
    return {"p%d" % p: ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)] for p in (50, 95, 99)}


class _RecordedRunner:
    def __init__(self, cases, results):
        self.rows = iter(zip(cases, results))

    def predict(self, state, questions, model=None):
        case, result = next(self.rows)
        if _sha([state, questions]) != _sha([case["state"], case["questions"]]):
            raise EvalError("recorded evaluation changed input order")
        return deepcopy(result)


def _quality(cases, results):
    labelled = [evals.Example.from_dict(case) for case in cases]
    report = evals.evaluate(_RecordedRunner(cases, results), evals.Dataset(labelled),
                            evaluators=evals.default_evaluators() + [evals.ScoreWithin(0.25)])
    names = {evaluator.name for evaluator in evals.default_evaluators() + [evals.ScoreWithin(0.25)]}
    return {name: value for name, value in report.overall.items() if name in names}


def _observe(runner, cases, batch_size, mode, concurrency=1):
    groups = {}
    for case in cases:
        groups.setdefault(_sha(case["questions"]), []).append(case)
    found, waits, work = {}, [], []
    for group in groups.values():
        group = list(reversed(group)) if mode == "reversed" else group
        capacity = batch_size - 1 if mode == "neighbour" else batch_size
        for offset in range(0, len(group), capacity):
            chunk = group[offset:offset + capacity]
            states = [deepcopy(case["state"]) for case in chunk]
            if mode == "neighbour":
                states.append("Unrelated qualification neighbour: archived weather observations without any task evidence.")
            work.append((chunk, states))

    def invoke(item):
        chunk, states = item
        begun = time.perf_counter()
        if batch_size == 1:
            results = [runner.predict(states[0], deepcopy(chunk[0]["questions"]))]
        else:
            if not callable(getattr(runner, "predict_batch", None)):
                raise EvalError("runner does not support the required batch entry point")
            results = list(runner.predict_batch(states, deepcopy(chunk[0]["questions"]), batch_size=batch_size))
        elapsed = (time.perf_counter() - begun) * 1000
        if len(results) != len(states):
            raise EvalError("runner returned %d results for %d states" % (len(results), len(states)))
        for result in results:
            _answers(result, chunk[0]["questions"])
        evidence = deepcopy(getattr(getattr(runner, "_local", None), "forwards", []))
        return chunk, results, elapsed, {"states": len(states), "target_states": len(chunk),
                                        "elapsed_ms": elapsed, "forwards": evidence}

    begun = time.perf_counter()
    if concurrency == 1:
        completed = list(map(invoke, work))
    else:
        # This is a closed-loop load probe. Only `concurrency` callers enter the
        # runtime at once; executor/admission waiting is not request latency.
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="veya-qualification") as pool:
            completed = list(pool.map(invoke, work))
    elapsed_total = (time.perf_counter() - begun) * 1000
    for chunk, results, elapsed, _ in completed:
        for case, result in zip(chunk, results):
            found[case["id"]] = result
            waits.append(elapsed)
    results = [found[case["id"]] for case in cases]
    observations = [{"id": case["id"], "answers": _answers(result, case["questions"])}
                    for case, result in zip(cases, results)]
    return {"observations": observations, "latency_ms": _percentiles(waits),
            "latencies_ms": waits, "metrics": _quality(cases, results),
            "elapsed_ms": elapsed_total, "requests_per_second": len(cases) * 1000 / elapsed_total,
            "execution": [execution for _, _, _, execution in completed]}


def _compare(reference, candidate):
    if len(reference) != len(candidate):
        raise EvalError("qualification observation cardinality differs from the reference")
    flips, labels, max_probability, max_score, max_confidence, max_action = 0, 0, 0.0, 0.0, 0.0, 0.0
    flag_flips, flag_comparisons = 0, 0
    for old, new in zip(reference, candidate):
        if old["id"] != new["id"] or set(old["answers"]) != set(new["answers"]):
            raise EvalError("qualification observations do not align")
        for qid, expected in old["answers"].items():
            actual = new["answers"][qid]
            if expected["answer"]["type"] != actual["answer"]["type"]:
                raise EvalError("qualification observation answer types differ")
            if set(expected["probabilities"]) != set(actual["probabilities"]):
                raise EvalError("qualification observation probability labels differ")
            if set(expected["operational"]) != set(actual["operational"]):
                raise EvalError("qualification confidence/action field presence changed")
            if set(expected["flags"]) != set(actual["flags"]):
                raise EvalError("qualification abstention/review field presence changed")
            if expected["flags"]:
                flag_comparisons += 1
                flag_flips += expected["flags"] != actual["flags"]
            for field, value in expected["operational"].items():
                drift = abs(value - actual["operational"][field])
                if field == "act_probability":
                    max_action = max(max_action, drift)
                else:
                    max_confidence = max(max_confidence, drift)
            if expected["label"] is not None:
                labels += 1
                flips += expected["label"] != actual["label"]
            max_probability = max(max_probability, *(abs(p - actual["probabilities"][key])
                                                      for key, p in expected["probabilities"].items()))
            if expected["score"] is not None:
                max_score = max(max_score, abs(expected["score"] - actual["score"]))
    return {"label_flips": flips, "labelled_comparisons": labels,
            "label_flip_rate": flips / labels if labels else 0.0,
            "max_probability_drift": max_probability, "max_score_drift": max_score,
            "max_confidence_drift": max_confidence, "max_action_drift": max_action,
            "operational_flag_flips": flag_flips, "operational_flag_comparisons": flag_comparisons,
            "operational_flag_flip_rate": flag_flips / flag_comparisons if flag_comparisons else 0.0}


def _saved_observations(run):
    observations = run.get("observations") if isinstance(run, dict) else None
    if not isinstance(observations, list) or not observations:
        raise EvalError("saved qualification run has no observations")
    ids = set()
    for row in observations:
        if not isinstance(row, dict) or not isinstance(row.get("answers"), dict) or not row["answers"]:
            raise EvalError("saved qualification observations must contain answers")
        cid = _name(row.get("id"), "observation.id")
        if cid in ids:
            raise EvalError("saved qualification observations repeat a case id")
        ids.add(cid)
        questions, answers = {}, {}
        for qid, observation in row["answers"].items():
            if not isinstance(observation, dict) or not isinstance(observation.get("answer"), dict):
                raise EvalError("saved qualification answer is malformed")
            raw, probabilities = observation["answer"], observation.get("probabilities")
            if not isinstance(probabilities, dict):
                raise EvalError("saved qualification answer is missing probabilities")
            kind = raw.get("type")
            if kind not in ("choice", "noul", "score"):
                raise EvalError("saved qualification answer has an unknown type")
            questions[qid] = {"type": kind, "criteria": {key: None for key in probabilities} if kind == "choice"
                              else [None] * len(probabilities)}
            answers[qid] = raw
        if _answers({"answers": answers}, questions) != row["answers"]:
            raise EvalError("saved qualification observations disagree with their raw answers")
    return observations


def _validate_saved_report(report, *, diagnostic=False):
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise EvalError("expected a saved qualification report")
    _encoded(report)
    limits = _limits(report.get("limits"))
    if report.get("production_qualified") is True and (
            report.get("scope") != "local-artifact-matrix" or limits["min_accuracy"] is None
            or limits["min_boundary_cases"] < 1):
        raise EvalError("saved production qualification has no explicit release quality policy")
    repeats = _integer(report.get("repeats"), "saved repeats", maximum=100)
    sizes, concurrencies = report.get("batch_sizes"), report.get("concurrencies")
    for name, values in (("batch_sizes", sizes), ("concurrencies", concurrencies)):
        if not isinstance(values, list) or not values:
            raise EvalError("saved %s must be a nonempty list" % name)
        for value in values:
            _integer(value, "saved " + name, maximum=64)
        if values != sorted(set(values)) or values[0] != 1:
            raise EvalError("saved %s must be unique, ordered and include 1" % name)
    expected_profiles = [(repeat, size, mode, concurrency)
                         for repeat in range(repeats) for size in sizes
                         for mode in (("ordered",) if size == 1 else ("ordered", "reversed", "neighbour"))
                         for concurrency in concurrencies]
    _name(report.get("corpus_id"), "report.corpus_id")
    digest = report.get("corpus_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise EvalError("saved qualification report has no valid corpus identity")
    if not diagnostic and report.get("production_qualified") is not True:
        raise EvalError("release comparison requires a production-qualified saved report; use diagnostic mode otherwise")
    if report.get("passed") is not True or report.get("failures") != []:
        raise EvalError("a failed qualification report cannot be the release reference")
    cells = report.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise EvalError("saved qualification report has no cells")
    if (report.get("reference") not in cells or not isinstance(cells[report["reference"]], dict)
            or cells[report["reference"]].get("status") != "passed"):
        raise EvalError("saved qualification reference did not complete")
    reference_runs = cells[report["reference"]].get("runs")
    if not isinstance(reference_runs, list) or not reference_runs:
        raise EvalError("saved qualification reference has no completed observations")
    canonical = _saved_observations(reference_runs[0])
    case_ids = [row["id"] for row in canonical]
    coverage = report.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("evaluation_cases") != len(canonical):
        raise EvalError("saved qualification case coverage disagrees with its observations")
    measured = [{"id": row["id"], "qid": qid, "margin": answer["margin"]}
                for row in canonical for qid, answer in row["answers"].items()
                if answer["margin"] <= limits["boundary_margin"]]
    if (coverage.get("boundary_answers") != len(measured)
            or coverage.get("boundary_cases") != len({row["id"] for row in measured})
            or coverage.get("boundary_observations") != measured):
        raise EvalError("saved qualification boundary coverage disagrees with its observations")
    if coverage["boundary_cases"] < limits["min_boundary_cases"]:
        raise EvalError("saved qualification does not satisfy its boundary coverage gate")
    reference_metrics = reference_runs[0].get("metrics")
    evals._metric_map(reference_metrics, "saved reference", allow_empty=True)
    for cell in cells.values():
        if not isinstance(cell, dict) or cell.get("status") not in ("passed", "skipped"):
            raise EvalError("saved qualification report has incomplete cells")
        if cell["status"] == "skipped":
            continue
        if not isinstance(cell.get("identity"), dict) or not isinstance(cell.get("runs"), list) or not cell["runs"]:
            raise EvalError("saved qualification cell has no identity or completed runs")
        if "decision_policy" in cell["identity"]:
            policy = cell["identity"]["decision_policy"]
            if not isinstance(policy, str) or policy not in DECISION_POLICIES:
                raise EvalError("saved qualification cell has an unknown decision policy")
        if not diagnostic:
            identity = cell["identity"]
            if identity.get("verification") != "local artifacts and runtime":
                raise EvalError("saved release cell has no verified runtime identity")
            if (identity.get("backend") not in ("eager", "compile", "tilelang", "onnx")
                    or not isinstance(identity.get("device"), str)
                    or identity.get("precision") not in ("float32", "float16", "bfloat16", "graph")):
                raise EvalError("saved release cell has no backend/device/precision identity")
            artifacts, fingerprint = identity.get("artifacts"), identity.get("model_fingerprint")
            if not isinstance(artifacts, dict) or not artifacts or not isinstance(fingerprint, list) or len(fingerprint) < 6:
                raise EvalError("saved release cell has no artifact/calibration identity")
            for digest in artifacts.values():
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise EvalError("saved release cell has an invalid artifact digest")
        for run in cell["runs"]:
            observations = _saved_observations(run)
            if [row["id"] for row in observations] != case_ids:
                raise EvalError("saved qualification runs have different case coverage or ordering")
            for original, actual in zip(canonical, observations):
                if set(original["answers"]) != set(actual["answers"]):
                    raise EvalError("saved qualification runs have different question coverage")
                for qid, answer in original["answers"].items():
                    other = actual["answers"][qid]
                    if (answer["answer"]["type"] != other["answer"]["type"]
                            or set(answer["probabilities"]) != set(other["probabilities"])):
                        raise EvalError("saved qualification answer schemas differ")
            _integer(run.get("batch_size"), "saved batch size", maximum=64)
            _integer(run.get("repeat"), "saved repeat", minimum=0, maximum=99)
            _integer(run.get("concurrency"), "saved concurrency", maximum=64)
            if run.get("mode") not in ("ordered", "reversed", "neighbour"):
                raise EvalError("saved run has an invalid ordering mode")
            evals._metric_map(run.get("metrics"), "saved qualification", allow_empty=True)
            for value in run["metrics"].values():
                _number(value, "saved metric")
            comparison = _compare(canonical, observations)
            if run.get("comparison") != comparison:
                raise EvalError("saved qualification drift summary disagrees with its observations")
            for metric, key in _DRIFT_LIMITS.items():
                limit = limits[key]
                if comparison[metric] > limit:
                    raise EvalError("saved qualification violates its %s gate" % metric)
            accuracies = {key: value for key, value in run["metrics"].items()
                          if key.endswith("accuracy") or key.startswith("score_within_")}
            if limits["min_accuracy"] is not None and (
                    not accuracies or any(value < limits["min_accuracy"] for value in accuracies.values())):
                raise EvalError("saved qualification violates its minimum accuracy gate")
            for metric, old in reference_metrics.items():
                if metric.endswith("accuracy") or metric.startswith("score_within_"):
                    new = run["metrics"].get(metric)
                    if new is None or old - new > limits["max_accuracy_drop"]:
                        raise EvalError("saved qualification violates its accuracy regression gate")
            latency = run.get("latency_ms")
            if not isinstance(latency, dict) or set(latency) != {"p50", "p95", "p99"}:
                raise EvalError("saved qualification run has no latency quantiles")
            for value in latency.values():
                _number(value, "saved latency", 0)
            samples = run.get("latencies_ms")
            if not isinstance(samples, list) or len(samples) != len(observations):
                raise EvalError("saved latency samples do not match completed case coverage")
            for value in samples:
                _number(value, "saved latency sample", 0)
            if _percentiles(samples) != latency:
                raise EvalError("saved latency quantiles disagree with their raw samples")
            if not latency["p50"] <= latency["p95"] <= latency["p99"]:
                raise EvalError("saved latency quantiles are not ordered")
            for p in (95, 99):
                limit = limits["max_p%d_ms" % p]
                if limit is not None and latency["p%d" % p] > limit:
                    raise EvalError("saved qualification violates its latency gate")
        first = cell["runs"][0]
        if (first["batch_size"], first["mode"], first["repeat"], first["concurrency"]) != (1, "ordered", 0, 1):
            raise EvalError("saved qualification baseline must start with a serial observation")
        profiles = [(run["repeat"], run["batch_size"], run["mode"], run["concurrency"]) for run in cell["runs"]]
        if profiles != expected_profiles:
            raise EvalError("saved qualification execution profiles are incomplete")
    return report


def compare_reports(candidate, baseline, *, limits=None, allow_unverified_identity=False):
    """Compare every current observation with a reviewed saved serial baseline.

    A fixed corpus and matching artifact/backend/precision/calibration identities
    are required. Source versions may differ: detecting their output changes is
    the purpose of the comparison. Custom-runner reports need diagnostic opt-in.
    """
    if type(allow_unverified_identity) is not bool:
        raise EvalError("allow_unverified_identity must be a boolean")
    candidate = _validate_saved_report(candidate, diagnostic=allow_unverified_identity)
    baseline = _validate_saved_report(baseline, diagnostic=allow_unverified_identity)
    explicit_limits = limits is not None
    limits = _limits(baseline.get("limits") if limits is None else limits)
    report = {"schema": "veya-qualification-comparison/1", "passed": True, "failures": [], "cells": {},
              "baseline_sha256": _sha(baseline), "candidate_sha256": _sha(candidate), "limits": limits}
    for key in ("corpus_id", "corpus_sha256", "reference", "batch_sizes", "concurrencies", "repeats"):
        if candidate.get(key) != baseline.get(key):
            report["failures"].append("incomparable %s" % key)
    if not explicit_limits and candidate.get("limits") != baseline.get("limits"):
        report["failures"].append("qualification gate policy differs from the reviewed baseline")
    completed = lambda document: {name for name, cell in document["cells"].items() if cell["status"] == "passed"}
    if completed(candidate) != completed(baseline):
        report["failures"].append("completed backend coverage differs from the reviewed baseline")
    for name in sorted(completed(candidate) & completed(baseline)):
        current_cell, old_cell = candidate["cells"][name], baseline["cells"][name]
        old_identity, identity = old_cell["identity"], current_cell["identity"]
        for key in ("verification", "backend", "device", "precision", "artifacts", "providers", "math_settings",
                    "threads", "intra_op_threads", "inter_op_threads", "decision_policy"):
            if old_identity.get(key) != identity.get(key):
                report["failures"].append("%s: incomparable %s" % (name, key))
        old_fingerprint, fingerprint = old_identity.get("model_fingerprint"), identity.get("model_fingerprint")
        if (isinstance(old_fingerprint, list) and isinstance(fingerprint, list)
                and old_fingerprint[5:] != fingerprint[5:]):
            report["failures"].append("%s: calibration or inference policy differs" % name)
        reference = old_cell["runs"][0]
        old_profiles = [(r["batch_size"], r["mode"], r["repeat"], r.get("concurrency", 1)) for r in old_cell["runs"]]
        profiles = [(r["batch_size"], r["mode"], r["repeat"], r.get("concurrency", 1)) for r in current_cell["runs"]]
        if profiles != old_profiles:
            report["failures"].append("%s: execution profiles differ" % name)
        deltas = []
        for run in current_cell["runs"]:
            comparison = _compare(reference["observations"], run["observations"])
            deltas.append(comparison)
            for metric, key in _DRIFT_LIMITS.items():
                limit = limits[key]
                if comparison[metric] > limit:
                    report["failures"].append("%s: saved-baseline %s=%g exceeds %g" % (name, metric, comparison[metric], limit))
            for metric, old in reference["metrics"].items():
                if metric.endswith("accuracy") or metric.startswith("score_within_"):
                    new = run["metrics"].get(metric)
                    if new is None or old - new > limits["max_accuracy_drop"]:
                        report["failures"].append("%s: saved-baseline %s regressed" % (name, metric))
            for p in (95, 99):
                limit = limits["max_p%d_ms" % p]
                if limit is not None and run["latency_ms"]["p%d" % p] > limit:
                    report["failures"].append("%s: saved-baseline p%d latency exceeds its limit" % (name, p))
        report["cells"][name] = deltas
    report["passed"] = not report["failures"]
    return report


def qualify(corpus, runners, *, reference, batch_sizes=(1, 2, 4), repeats=2,
            limits=None, allow_unverified_identity=False, concurrencies=(1,)):
    """Qualify live local runners across serial/batch/order/neighbour variations.

    Verified local adapters are built by ``run_matrix``. Custom runners require
    explicit diagnostic opt-in; their report cannot claim production qualification.
    The reference is the first serial observation, not ground truth. Accuracy is
    measured separately using only the corpus's explicit expected labels.
    """
    return _qualify(corpus, runners, reference=reference, batch_sizes=batch_sizes, repeats=repeats,
                    limits=limits, allow_unverified_identity=allow_unverified_identity,
                    concurrencies=concurrencies)


def _qualify(corpus, runners, *, reference, batch_sizes, repeats, limits,
             allow_unverified_identity=False, baseline_run=None, concurrencies=(1,)):
    corpus, limits = validate_corpus(corpus), _limits(limits)
    if (not isinstance(runners, dict) or not runners
            or (reference not in runners and baseline_run is None)):
        raise EvalError("runners must name a nonempty matrix containing the reference")
    if type(allow_unverified_identity) is not bool:
        raise EvalError("allow_unverified_identity must be a boolean")
    repeats = _integer(repeats, "repeats", maximum=100)
    if not isinstance(batch_sizes, (list, tuple)) or not batch_sizes:
        raise EvalError("batch_sizes must be a nonempty sequence")
    sizes = sorted({_integer(size, "batch size", maximum=64) for size in batch_sizes} | {1})
    if not isinstance(concurrencies, (list, tuple)) or not concurrencies:
        raise EvalError("concurrencies must be a nonempty sequence")
    concurrency_profiles = sorted({_integer(value, "concurrency", maximum=64) for value in concurrencies} | {1})
    verified = all(type(runner) is _LocalRunner for runner in runners.values())
    if not verified and not allow_unverified_identity:
        raise EvalError("unverified runner identity; use run_matrix for release qualification or explicitly opt into diagnostics")
    cases = [case for case in corpus["cases"] if case["split"] == "evaluation"]
    report = {"schema": REPORT_SCHEMA, "corpus_id": corpus["id"], "corpus_sha256": _sha(corpus),
              "reference": reference, "limits": limits, "batch_sizes": sizes, "repeats": repeats,
              "concurrencies": concurrency_profiles,
              "scope": "local-artifact-matrix" if verified else "diagnostic",
              "production_qualified": False, "passed": True, "failures": [], "cells": {},
              "coverage": {"evaluation_cases": len(cases), "families": len({c["family"] for c in cases}),
                           "labelled_answers": sum(len(c["expected"]) for c in cases)}}
    baseline = deepcopy(baseline_run["observations"]) if baseline_run else None
    baseline_metrics = deepcopy(baseline_run["metrics"]) if baseline_run else None
    for name in ([reference] if reference in runners else []) + [key for key in runners if key != reference]:
        _name(name, "cell name")
        runner = runners[name]
        cell = {"status": "passed", "identity": runner.identity if type(runner) is _LocalRunner else
                {"verification": "unverified custom runner"}, "runs": []}
        report["cells"][name] = cell
        try:
            profiles = [(repeat, size, mode, concurrency)
                        for repeat in range(repeats) for size in sizes
                        for mode in (("ordered",) if size == 1 else ("ordered", "reversed", "neighbour"))
                        for concurrency in concurrency_profiles]
            for repeat, size, mode, concurrency in profiles:
                run = _observe(runner, cases, size, mode, concurrency)
                run.update(batch_size=size, mode=mode, repeat=repeat, concurrency=concurrency)
                if baseline is None:
                    baseline, baseline_metrics = run["observations"], run["metrics"]
                comparison = run["comparison"] = _compare(baseline, run["observations"])
                failures = []
                for metric, key in _DRIFT_LIMITS.items():
                    limit = limits[key]
                    if comparison[metric] > limit:
                        failures.append("%s=%g exceeds %g" % (metric, comparison[metric], limit))
                accuracies = {key: value for key, value in run["metrics"].items()
                              if key.endswith("accuracy") or key.startswith("score_within_")}
                if limits["min_accuracy"] is not None:
                    if not accuracies:
                        failures.append("min_accuracy has no labelled accuracy metric")
                    failures.extend("%s=%g is below %g" % (metric, value, limits["min_accuracy"])
                                    for metric, value in accuracies.items() if value < limits["min_accuracy"])
                for metric, value in baseline_metrics.items():
                    if metric.endswith("accuracy") or metric.startswith("score_within_"):
                        if metric not in run["metrics"] or value - run["metrics"][metric] > limits["max_accuracy_drop"]:
                            failures.append("%s regressed beyond max_accuracy_drop" % metric)
                for p in (95, 99):
                    limit = limits["max_p%d_ms" % p]
                    if limit is not None and run["latency_ms"]["p%d" % p] > limit:
                        failures.append("p%d latency exceeds %g ms" % (p, limit))
                run["failures"] = failures
                cell["runs"].append(run)
                if failures:
                    cell["status"] = "failed"
                    report["failures"].extend("%s repeat=%d batch=%d concurrency=%d %s: %s" %
                                              (name, repeat, size, concurrency, mode, failure) for failure in failures)
        except Exception as exc:
            cell["status"], cell["error"] = "failed", "%s: %s" % (type(exc).__name__, exc)
            report["failures"].append("%s could not complete: %s" % (name, cell["error"]))
        if name == reference and baseline is None:
            break
    boundary = [{"id": row["id"], "qid": qid, "margin": answer["margin"]}
                for row in baseline or [] for qid, answer in row["answers"].items()
                if answer["margin"] <= limits["boundary_margin"]]
    boundary_cases = len({row["id"] for row in boundary})
    report["coverage"].update(boundary_answers=len(boundary), boundary_cases=boundary_cases,
                              boundary_observations=boundary)
    if boundary_cases < limits["min_boundary_cases"]:
        report["failures"].append("reference has %d measured boundary cases; need %d" %
                                  (boundary_cases, limits["min_boundary_cases"]))
    report["passed"] = not report["failures"]
    report["production_qualified"] = (report["passed"] and verified and limits["min_accuracy"] is not None
                                       and limits["min_boundary_cases"] >= 1)
    report["measurement"] = {"latency": "whole completed local call, unsplit per target request; includes local inference queue waiting",
                             "load": "bounded closed-loop callers; threadpool admission waiting excluded; no HTTP SLO claim",
                             "warmth": "model load excluded; first call can include cold operator initialization",
                             "tail_warning": "small samples do not certify production tail latency",
                             "reference": "first serial run; agreement is separate from correctness",
                             "boundary": "measured top-two probability gap; score drift is gated separately"}
    return report


class _Unavailable(RuntimeError):
    pass


def _file_sha(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise EvalError("artifact changed while hashing: %s" % path)
    return digest.hexdigest()


def _source_identity():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)).replace("\\", "/"): _file_sha(path)
            for path in sorted(root.rglob("*.py"))}


def _artifact_identity(checkpoint, graph=None):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir() or not (checkpoint / "rl_agent_config.json").is_file():
        raise EvalError("checkpoint must be an existing local Veya checkpoint directory")
    files = set(checkpoint.rglob("*.json")) | set(checkpoint.glob("*.safetensors"))
    files.update(file for file in (checkpoint / "tokenizer").rglob("*") if file.is_file())
    if not files or (graph is None and not any(file.suffix == ".safetensors" for file in files)):
        raise EvalError("local checkpoint weights are missing")
    result = {str(file.relative_to(checkpoint)).replace("\\", "/"): _file_sha(file)
              for file in sorted(files) if file.is_file()}
    if graph is not None:
        graph = Path(graph).resolve()
        if not graph.is_file():
            raise EvalError("ONNX graph must be an existing local file")
        # Use the same schema walk as replay, including sparse and nested tensors.
        from .replay import _onnx_artifacts
        try:
            files = _onnx_artifacts(graph)
        except ImportError as exc:
            raise _Unavailable("ONNX artifact verification requires the onnx extra") from exc
        except (OSError, ValueError) as exc:
            raise EvalError("cannot verify ONNX artifacts: %s" % exc) from exc
        for path in sorted(map(Path, files)):
            key = "onnx/" + graph.name if path == graph else "onnx-external/" + path.relative_to(graph.parent).as_posix()
            result[key] = _file_sha(path)
    return result


class _LocalRunner:
    def __init__(self, agent, identity):
        self.agent, self.identity = agent, identity
        self._witness = None
        self._forwards = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def _check(self):
        from .hooks import compose_hooks
        if self.agent is None:
            raise EvalError("qualification cell is closed")
        policy = getattr(self.agent, "decision_policy", "canonical")
        if (not isinstance(policy, str) or policy not in DECISION_POLICIES
                or policy != self.identity.get("decision_policy", "canonical")):
            raise EvalError("decision policy changed during qualification")
        if compose_hooks(getattr(self.agent, "hooks", ())):
            raise EvalError("qualification requires an unhooked, uncached Agent")
        if "model_fingerprint" in self.identity:
            from .consistency import _fingerprint
            if _fingerprint(self.agent) != self.identity["model_fingerprint"]:
                raise EvalError("model configuration or calibration changed during qualification")
        if self.identity["backend"] != "onnx":
            if "math_settings" in self.identity and _torch_math_settings(self.identity["device"]) != self.identity["math_settings"]:
                raise EvalError("numerical execution policy changed during qualification")
            precision = str(self.agent.dtype).replace("torch.", "") if self.agent.amp_enabled else "float32"
            actual = (str(self.agent.device), self.agent.backend, precision)
            expected = (self.identity["device"], self.identity["backend"], self.identity["precision"])
            if actual != expected:
                raise EvalError("runtime fallback changed qualification identity: %r != %r" % (actual, expected))
        elif self.agent.session.get_providers() != self.identity["providers"]:
            raise EvalError("ONNX execution providers changed during qualification")

    def _call(self, method, *args, **kwargs):
        with self._lock:
            self._check()
            self._forwards = []
            result = method(*args, **kwargs)
            self._check()
            if self._witness is not None and not self._forwards:
                raise EvalError("the requested backend did not expose verifiable forward execution")
            self._local.forwards = deepcopy(self._forwards)
            return deepcopy(result)

    def predict(self, state, questions, model=None):
        return self._call(self.agent.predict, state, questions)

    def predict_batch(self, states, questions, model=None, batch_size=None):
        return self._call(self.agent.predict_batch, states, questions, batch_size=batch_size)

    def close(self):
        if self._witness is not None:
            self._witness.remove()
            self._witness = None
        agent, self.agent = self.agent, None
        if agent is not None:
            _dispose_agent(agent)


def _dispose_agent(agent):
    """Break backend/model cycles before the next cell can allocate its weights."""
    try:
        deaccelerate = getattr(agent, "deaccelerate", None)
        try:
            if deaccelerate:
                deaccelerate()
        finally:
            close = getattr(agent, "close", None)
            exit_context = getattr(agent, "__exit__", None)
            if close:
                close()
            elif exit_context:
                exit_context(None, None, None)
            elif hasattr(agent, "session"):
                agent.session = None
            if str(getattr(agent, "device", "")).startswith("mps"):
                import torch
                torch.mps.empty_cache()
    finally:
        import gc
        gc.collect()


def _torch_math_settings(device):
    import torch
    modern = getattr(torch.backends, "fp32_precision", None)
    settings = {"float32_matmul_precision": "per-backend" if modern is not None else torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if modern is not None:
        # The old global getter/allow_tf32 properties can raise when valid new
        # per-backend policies differ. Inspect one API family, never mix them.
        settings["global_fp32_precision"] = modern
    if device.startswith("cuda"):
        if modern is not None:
            settings["cuda_matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
            settings["cudnn_fp32_precision"] = torch.backends.cudnn.fp32_precision
            for operation in ("conv", "rnn"):
                settings["cudnn_" + operation + "_fp32_precision"] = getattr(getattr(torch.backends.cudnn, operation, None), "fp32_precision", None)
        else:
            settings["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            settings["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        for field in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
            settings["cuda_matmul_" + field] = getattr(torch.backends.cuda.matmul, field, None)
        settings["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        settings["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    elif device == "cpu":
        settings["mkldnn_enabled"] = torch.backends.mkldnn.enabled
        if modern is not None:
            settings["mkldnn_fp32_precision"] = getattr(torch.backends.mkldnn, "fp32_precision", None)
            for operation in ("matmul", "conv", "rnn"):
                settings["mkldnn_" + operation + "_fp32_precision"] = getattr(getattr(torch.backends.mkldnn, operation, None), "fp32_precision", None)
    return settings


def _build_cell(spec):
    policy_args = {}
    if "decision_policy" in spec:
        policy = spec["decision_policy"]
        if not isinstance(policy, str) or policy not in DECISION_POLICIES:
            raise EvalError("cell.decision_policy must be canonical or context-balanced-v1")
        policy_args["decision_policy"] = policy
    graph = spec.get("onnx_path")
    artifacts = _artifact_identity(spec["checkpoint"], graph)
    checkpoint = str(Path(spec["checkpoint"]).resolve())
    backend, device, precision = spec["backend"], spec["device"], spec["precision"]
    if backend == "onnx":
        from .onnx_agent import _NativeONNXAgent
        try:
            import onnxruntime
            agent = _NativeONNXAgent(checkpoint, onnx_path=str(Path(graph).resolve()),
                                     performance_mode="default", **policy_args)
        except ImportError as exc:
            raise _Unavailable("ONNX runtime dependencies are unavailable: %s" % exc) from exc
        identity = {"verification": "local artifacts and runtime", "artifacts": artifacts,
                    "backend": "onnx", "device": "cpu", "precision": "graph",
                    "providers": agent.session.get_providers(), "onnxruntime": onnxruntime.__version__,
                    "intra_op_threads": agent.session.get_session_options().intra_op_num_threads,
                    "inter_op_threads": agent.session.get_session_options().inter_op_num_threads}
    else:
        import torch
        from .agent import Agent
        from .backends.base import BackendUnavailable
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise _Unavailable("CUDA is unavailable")
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise _Unavailable("MPS is unavailable")
        if device == "cuda":
            device = "cuda:%d" % torch.cuda.current_device()
        elif device == "mps":
            device = "mps:0"
        env_names = ("VEYA_CPU_AMP", "VEYA_CUDA_AMP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        held = {name: os.environ.get(name) for name in env_names}
        agent = None
        try:
            os.environ.update(VEYA_CPU_AMP="bf16" if precision == "bf16" else "fp32",
                              VEYA_CUDA_AMP=precision, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            agent = Agent(checkpoint, device=device, backend="eager", performance_mode="default", **policy_args)
            if device.startswith("mps"):
                agent.amp_enabled, agent.dtype = precision == "fp16", torch.float16 if precision == "fp16" else torch.float32
                agent.mps_amp_min_rows = 1
            if backend != "eager":
                agent.set_backend(backend, strict=True)
        except BaseException as exc:
            if agent is not None:
                try:
                    _dispose_agent(agent)
                except BaseException as cleanup:
                    raise exc from cleanup
            if isinstance(exc, BackendUnavailable):
                raise _Unavailable("requested backend is unavailable: %s" % exc) from exc
            raise
        finally:
            for name, value in held.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        identity = {"verification": "local artifacts and runtime", "artifacts": artifacts,
                    "backend": backend, "device": str(torch.device(device)),
                    "precision": {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[precision],
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "math_settings": _torch_math_settings(device)}
    runner = _LocalRunner(agent, identity)
    effective_policy = getattr(agent, "decision_policy", "canonical")
    if effective_policy != "canonical":
        identity["decision_policy"] = effective_policy
    from .consistency import _fingerprint
    identity["model_fingerprint"] = deepcopy(_fingerprint(agent))
    if backend != "onnx":
        def witness(module, args, kwargs):
            tokens = kwargs.get("input_ids")
            if tokens is None and args:
                tokens = args[0]
            if tokens is None:
                raise EvalError("cannot verify forward input device")
            actual_device = str(tokens.device)
            kind = tokens.device.type
            try:
                enabled = torch.is_autocast_enabled(kind)
                dtype = torch.get_autocast_dtype(kind) if enabled else torch.float32
            except TypeError:  # torch 2.0's device-specific autocast inspection
                enabled = torch.is_autocast_cpu_enabled() if kind == "cpu" else torch.is_autocast_enabled()
                dtype = (torch.get_autocast_cpu_dtype() if kind == "cpu" else torch.get_autocast_gpu_dtype()) if enabled else torch.float32
            actual_precision = str(dtype).replace("torch.", "")
            if (actual_device, actual_precision) != (identity["device"], identity["precision"]):
                raise EvalError("forward executed with an unexpected device or precision: %s/%s" %
                                (actual_device, actual_precision))
            runner._forwards.append({"device": actual_device, "precision": actual_precision,
                                     "question_rows": int(tokens.shape[0])})
        runner._witness = agent.model.register_forward_pre_hook(witness, with_kwargs=True)
    try:
        expected_policy = resolve_decision_policy(policy_args.get("decision_policy"), getattr(agent, "cfg", None))
        if effective_policy != expected_policy:
            raise EvalError("runtime did not activate the requested decision policy")
        runner._check()
        if _artifact_identity(spec["checkpoint"], graph) != artifacts:
            raise EvalError("checkpoint changed while the qualification cell loaded")
    except BaseException as exc:
        try:
            runner.close()
        except BaseException as cleanup:
            raise exc from cleanup
        raise
    return runner


def validate_matrix(document):
    """Validate the local-only deployment matrix before loading a checkpoint."""
    if not isinstance(document, dict) or document.get("schema") != MATRIX_SCHEMA:
        raise EvalError("expected matrix schema %r" % MATRIX_SCHEMA)
    _encoded(document)
    if set(document) - {"schema", "reference", "cells", "batch_sizes", "repeats", "limits", "concurrencies"}:
        raise EvalError("unknown qualification matrix field")
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise EvalError("matrix.cells must be a nonempty list")
    names = set()
    for spec in cells:
        if not isinstance(spec, dict) or set(spec) - {"name", "checkpoint", "backend", "device", "precision", "onnx_path", "required", "decision_policy"}:
            raise EvalError("invalid qualification matrix cell")
        name = _name(spec.get("name"), "cell.name")
        if name in names:
            raise EvalError("duplicate matrix cell name %r" % name)
        names.add(name)
        _name(spec.get("checkpoint"), "cell.checkpoint")
        backend, device, precision = spec.get("backend"), spec.get("device"), spec.get("precision")
        if backend not in ("eager", "compile", "tilelang", "onnx"):
            raise EvalError("cell.backend must be eager, compile, tilelang or onnx")
        if not isinstance(device, str) or not (device in ("cpu", "mps", "cuda") or
                                               (device.startswith("cuda:") and device[5:].isdigit())):
            raise EvalError("cell.device must explicitly name cpu, mps or a CUDA device")
        if type(spec.get("required", True)) is not bool:
            raise EvalError("cell.required must be a boolean")
        if "decision_policy" in spec and (not isinstance(spec["decision_policy"], str)
                                           or spec["decision_policy"] not in DECISION_POLICIES):
            raise EvalError("cell.decision_policy must be canonical or context-balanced-v1")
        if backend == "onnx":
            if device != "cpu" or precision != "graph" or not isinstance(spec.get("onnx_path"), str):
                raise EvalError("ONNX cells require cpu, precision='graph' and onnx_path")
        elif (precision not in ("fp32", "fp16", "bf16") or "onnx_path" in spec or
              (device == "cpu" and precision == "fp16") or (device == "mps" and precision == "bf16")):
            raise EvalError("unsupported explicit precision/device pair")
        if backend == "tilelang" and not device.startswith("cuda"):
            raise EvalError("tilelang qualification requires CUDA")
    if document.get("reference") not in names:
        raise EvalError("matrix.reference must name a cell")
    if not next(spec for spec in cells if spec["name"] == document["reference"]).get("required", True):
        raise EvalError("the reference cell must be required")
    _limits(document.get("limits"))
    _integer(document.get("repeats", 2), "repeats", maximum=100)
    sizes = document.get("batch_sizes", [1, 2, 4])
    if not isinstance(sizes, list) or not sizes:
        raise EvalError("batch_sizes must be a nonempty list")
    for size in sizes:
        _integer(size, "batch size", maximum=64)
    concurrencies = document.get("concurrencies", [1])
    if not isinstance(concurrencies, list) or not concurrencies:
        raise EvalError("concurrencies must be a nonempty list")
    for value in concurrencies:
        _integer(value, "concurrency", maximum=64)
    return deepcopy(document)


def run_matrix(corpus, matrix):
    """Run verified local cells sequentially and fail every unavailable required cell.

    Optional unavailable cells are explicitly skipped. They are excluded from the
    qualification scope and can never count as successful backend coverage.
    """
    corpus, matrix = validate_corpus(corpus), validate_matrix(matrix)
    initial_sources = _source_identity()
    reference = matrix["reference"]
    specs = sorted(matrix["cells"], key=lambda spec: spec["name"] != reference)
    report = baseline_run = None
    for spec in specs:
        runner = None
        cleanup_failed = False
        try:
            runner = _build_cell(spec)
            partial = _qualify(corpus, {spec["name"]: runner}, reference=reference,
                               batch_sizes=matrix.get("batch_sizes", [1, 2, 4]),
                               repeats=matrix.get("repeats", 2), limits=matrix.get("limits"),
                               baseline_run=baseline_run, concurrencies=matrix.get("concurrencies", [1]))
            if ("artifacts" in runner.identity and
                    _artifact_identity(spec["checkpoint"], spec.get("onnx_path")) != runner.identity["artifacts"]):
                raise EvalError("checkpoint artifacts changed during qualification")
            if spec["name"] == reference:
                report = partial
                runs = report["cells"][reference]["runs"]
                if runs:
                    baseline_run = runs[0]
            else:
                report["cells"][spec["name"]] = partial["cells"][spec["name"]]
                report["failures"].extend(partial["failures"])
        except Exception as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
            if report is None:
                report = {"schema": REPORT_SCHEMA, "corpus_id": corpus["id"], "corpus_sha256": _sha(corpus),
                          "reference": reference, "cells": {}, "failures": [], "coverage": {}}
            skipped = not spec.get("required", True) and isinstance(exc, _Unavailable)
            report["cells"][spec["name"]] = {"status": "skipped" if skipped else "failed", "error": failure}
            if not skipped:
                report["failures"].append("cell %s could not complete: %s" % (spec["name"], failure))
        finally:
            if runner is not None:
                try:
                    runner.close()
                except Exception as exc:
                    cleanup_failed = True
                    report["cells"][spec["name"]]["status"] = "failed"
                    report["failures"].append("cell %s cleanup failed: %s: %s" %
                                              (spec["name"], type(exc).__name__, exc))
        if baseline_run is None or cleanup_failed:
            break
    for spec in matrix["cells"]:
        if spec["name"] not in report["cells"]:
            report["cells"][spec["name"]] = {"status": "not_run", "error": "qualification stopped before this cell"}
    report["matrix"] = matrix
    report["matrix_sha256"] = _sha(matrix)
    report["runtime"] = {"python": platform.python_version(), "platform": platform.platform()}
    from importlib import metadata
    dependencies = {}
    for package in ("veya", "torch", "transformers", "safetensors", "numpy", "tokenizers", "onnxruntime", "onnx"):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = "not installed"
    report["runtime"]["dependencies"] = dependencies
    report["coverage"]["requested_cells"] = [spec["name"] for spec in matrix["cells"]]
    report["coverage"]["completed_cells"] = [name for name, cell in report["cells"].items() if cell["status"] == "passed"]
    report["coverage"]["skipped_cells"] = [name for name, cell in report["cells"].items() if cell["status"] == "skipped"]
    report["coverage"]["not_run_cells"] = [name for name, cell in report["cells"].items() if cell["status"] == "not_run"]
    final_sources = _source_identity()
    report["source_sha256"] = initial_sources
    report["source_changed_during_run"] = final_sources != initial_sources
    if report["source_changed_during_run"]:
        report["failures"].append("runtime source files changed during qualification; rerun after edits stop")
    report["passed"] = not report["failures"]
    limits = _limits(matrix.get("limits"))
    report["production_qualified"] = (report["passed"] and limits["min_accuracy"] is not None
                                       and limits["min_boundary_cases"] >= 1)
    _encoded(report)
    return report
