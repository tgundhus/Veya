"""A reviewed release snapshot must catch changes shared by every current cell."""
from copy import deepcopy

import pytest

from laya import qualification
from laya.qualification import compare_reports, qualify


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Allow this request?",
              "criteria": {"allow": "Authorized", "deny": "Unauthorized"}},
    "authorized": {"type": "noul", "instructions": "Is the request authorized?"},
    "priority": {"type": "score", "instructions": "Rate priority.", "criteria": ["Low", "Medium", "High"]},
}


def make_report(*, flip=False, probability_offset=0.0, score_offset=0.0, operational=None):
    document = {"schema": "veya-qualification-corpus/1", "id": "saved-release-test-v1", "cases": [
        {"id": "approved" if keep else "rejected", "family": "authorization-family", "split": "evaluation",
         "variant": "base" if keep else "counterfactual", "state": {"keep": keep},
         "questions": deepcopy(QUESTIONS),
         "expected": {"route": "allow" if keep else "deny", "authorized": keep, "priority": 1},
         "tags": ["boundary-candidate"]}
        for keep in (True, False)]}

    class Runner:
        def predict(self, state, questions, model=None):
            keep = state.get("keep", True) if isinstance(state, dict) else True
            keep = not keep if flip else keep
            p = (0.51 + probability_offset) if keep else (0.49 - probability_offset)
            result = {"answers": {
                "route": {"type": "choice", "choice": "allow" if keep else "deny",
                          "probabilities": {"allow": p, "deny": 1 - p}},
                "authorized": {"type": "noul", "noul": p},
                "priority": {"type": "score", "score": 1.0 + score_offset,
                             "probabilities": {"0": 0.0, "1": 1 - score_offset, "2": score_offset}}}}
            for answer in result["answers"].values():
                answer.update(confidence=0.51, answer_confidence=0.51, action={"act_probability": 0.51},
                              low_confidence=False, abstention="accepted")
                for field, value in (operational or {}).items():
                    if field == "act_probability":
                        answer["action"][field] = value
                    else:
                        answer[field] = value
            return result

        def predict_batch(self, states, questions, model=None, batch_size=None):
            return [self.predict(state, questions, model=model) for state in states]

    report = qualify(document, {"reference": Runner()}, reference="reference", batch_sizes=(1, 2), repeats=1,
                     allow_unverified_identity=True)
    # Diagnostic identities remain explicitly unverified. The extra metadata lets
    # these tests exercise comparability without claiming real artifact execution.
    report["cells"]["reference"]["identity"].update({
        "backend": "eager", "device": "cpu", "precision": "float32", "artifacts": {"weights": "a" * 64},
        "providers": ["CPUExecutionProvider"],
        "model_fingerprint": ["code-v1", "checkpoint", "revision", None, ["artifact"], 512, 192, [1.0, 1.0]],
    })
    report["source_sha256"] = {"qualification.py": "a" * 64}
    assert report["passed"] is True
    assert report["production_qualified"] is False
    return report


def compare(candidate, baseline, **kwargs):
    return compare_reports(candidate, baseline, allow_unverified_identity=True, **kwargs)


def assert_refused(candidate, baseline):
    try:
        result = compare(candidate, baseline)
    except ValueError:
        return
    assert result["passed"] is False
    assert result["failures"]


def test_unchanged_diagnostic_snapshot_passes_without_mutating_either_report():
    baseline = make_report()
    candidate = deepcopy(baseline)
    original = deepcopy(candidate)
    result = compare(candidate, baseline)
    assert result["schema"] == "veya-qualification-comparison/1"
    assert result["passed"] is True
    assert result["failures"] == []
    assert candidate == baseline == original


@pytest.mark.parametrize("change", ["labels", "probabilities", "score"])
def test_global_release_drift_fails_even_when_every_current_profile_agrees(change):
    baseline = make_report()
    candidate = make_report(flip=change == "labels", probability_offset=0.01 if change == "probabilities" else 0.0,
                            score_offset=0.01 if change == "score" else 0.0)
    assert candidate["passed"] is True
    assert all(run["comparison"]["label_flip_rate"] == 0
               for run in candidate["cells"]["reference"]["runs"])
    result = compare(candidate, baseline)
    assert result["passed"] is False
    assert any("saved-baseline" in failure for failure in result["failures"])


def test_raw_answer_tampering_cannot_disagree_with_saved_normalized_answers():
    baseline = make_report()
    candidate = deepcopy(baseline)
    answer = candidate["cells"]["reference"]["runs"][0]["observations"][0]["answers"]["route"]
    answer["answer"]["choice"] = "deny"
    with pytest.raises(ValueError, match="raw answers"):
        compare(candidate, baseline)


@pytest.mark.parametrize("change", ["label", "probabilities", "margin"])
def test_normalized_answer_tampering_is_rejected(change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    answer = candidate["cells"]["reference"]["runs"][0]["observations"][0]["answers"]["route"]
    if change == "label":
        answer["label"] = "deny"
    elif change == "probabilities":
        answer["probabilities"] = {"allow": 0.49, "deny": 0.51}
    else:
        answer["margin"] = 0.99
    with pytest.raises(ValueError, match="raw answers"):
        compare(candidate, baseline)


@pytest.mark.parametrize("change", ["missing_tail", "missing_qid", "duplicate_id", "missing_run_case"])
def test_missing_or_duplicate_saved_case_coverage_cannot_pass(change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    runs = candidate["cells"]["reference"]["runs"]
    if change == "missing_tail":
        for run in runs:
            run["observations"].pop()
    elif change == "missing_qid":
        for run in runs:
            run["observations"][0]["answers"].pop("authorized")
    elif change == "duplicate_id":
        runs[0]["observations"][1]["id"] = runs[0]["observations"][0]["id"]
    else:
        runs[-1]["observations"].pop()
    assert_refused(candidate, baseline)


@pytest.mark.parametrize("change", ["corpus_id", "corpus_sha256", "backend", "device", "precision", "artifacts",
    "providers", "calibration", "batch_sizes", "concurrencies", "repeats", "coverage", "profiles", "completed_cells"])
def test_identity_policy_and_profile_changes_are_not_comparable(change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    cell = candidate["cells"]["reference"]
    if change == "corpus_id":
        candidate["corpus_id"] = "different-corpus"
    elif change == "corpus_sha256":
        candidate["corpus_sha256"] = "b" * 64
    elif change in ("backend", "device", "precision"):
        cell["identity"][change] = "changed"
    elif change == "artifacts":
        cell["identity"]["artifacts"]["weights"] = "b" * 64
    elif change == "providers":
        cell["identity"]["providers"] = ["ChangedExecutionProvider"]
    elif change == "calibration":
        cell["identity"]["model_fingerprint"][-1] = [1.1, 1.0]
    elif change == "batch_sizes":
        candidate["batch_sizes"] = [1, 4]
    elif change == "concurrencies":
        candidate["concurrencies"] = [1, 2]
    elif change == "repeats":
        candidate["repeats"] = 2
    elif change == "coverage":
        candidate["coverage"]["evaluation_cases"] -= 1
    elif change == "profiles":
        cell["runs"][-1]["repeat"] = 1
    else:
        candidate["cells"]["extra-backend"] = deepcopy(cell)
    assert_refused(candidate, baseline)


def test_a_candidate_cannot_silently_loosen_the_reviewed_gate_policy():
    baseline = make_report()
    candidate = deepcopy(baseline)
    candidate["limits"]["max_probability_drift"] = 1.0
    result = compare(candidate, baseline)
    assert result["passed"] is False
    assert any("gate policy" in failure for failure in result["failures"])


def test_explicit_comparison_limits_are_recorded_when_deliberately_overridden():
    baseline = make_report()
    candidate = deepcopy(baseline)
    candidate["limits"]["max_probability_drift"] = 1.0
    result = compare(candidate, baseline, limits={"max_probability_drift": 1.0})
    assert result["passed"] is True
    assert result["limits"]["max_probability_drift"] == 1.0


def test_release_source_changes_are_allowed_for_a_regression_comparison():
    baseline = make_report()
    candidate = deepcopy(baseline)
    candidate["source_sha256"]["qualification.py"] = "b" * 64
    candidate["cells"]["reference"]["identity"]["model_fingerprint"][0] = "code-v2"
    assert compare(candidate, baseline)["passed"] is True


@pytest.mark.parametrize("change", ["wide_metric", "nonfinite_metric", "wide_latency", "wide_limit",
                                   "wide_probability", "bad_concurrency", "empty_observations", "bad_schema"])
def test_malformed_saved_numeric_data_is_a_validation_error(change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    run = candidate["cells"]["reference"]["runs"][0]
    if change == "wide_metric":
        run["metrics"]["choice_accuracy"] = 10 ** 400
    elif change == "nonfinite_metric":
        run["metrics"]["choice_accuracy"] = float("nan")
    elif change == "wide_latency":
        run["latency_ms"]["p95"] = 10 ** 400
    elif change == "wide_limit":
        candidate["limits"]["max_probability_drift"] = 10 ** 400
    elif change == "wide_probability":
        run["observations"][0]["answers"]["route"]["answer"]["probabilities"]["allow"] = 10 ** 400
    elif change == "bad_concurrency":
        run["concurrency"] = None
    elif change == "empty_observations":
        run["observations"] = []
    else:
        candidate["schema"] = "veya-qualification-report/999"
    with pytest.raises(ValueError):
        compare(candidate, baseline)


def test_unverified_saved_reports_require_an_explicit_diagnostic_opt_in():
    baseline = make_report()
    with pytest.raises(ValueError, match="production-qualified"):
        compare_reports(deepcopy(baseline), baseline)


def test_production_flag_alone_does_not_forge_a_verified_runtime_identity():
    baseline = make_report()
    baseline["production_qualified"] = True
    with pytest.raises(ValueError, match="quality policy|runtime identity"):
        compare_reports(deepcopy(baseline), baseline)


@pytest.mark.parametrize("change", ["missing_profile", "duplicate_profile", "unordered_quantiles"])
def test_partial_profiles_or_unordered_latency_quantiles_are_malformed(change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    runs = candidate["cells"]["reference"]["runs"]
    if change == "missing_profile":
        runs.pop()
    elif change == "duplicate_profile":
        runs.append(deepcopy(runs[-1]))
    else:
        runs[0]["latency_ms"] = {"p50": 10.0, "p95": 5.0, "p99": 20.0}
    with pytest.raises(ValueError):
        compare(candidate, baseline)


def test_a_new_zero_probability_option_cannot_hide_a_changed_answer_schema():
    baseline = make_report()
    candidate = deepcopy(baseline)
    for run in candidate["cells"]["reference"]["runs"]:
        for row in run["observations"]:
            answer = row["answers"]["route"]
            answer["answer"]["probabilities"]["surprise"] = 0.0
            answer["probabilities"]["surprise"] = 0.0
    with pytest.raises(ValueError, match="probability labels"):
        compare(candidate, baseline)


def test_a_forged_production_flag_does_not_override_failed_declared_quality_gates():
    forged = make_report()
    forged.update(production_qualified=True, scope="local-artifact-matrix")
    forged["limits"].update(min_accuracy=1.0, min_boundary_cases=1)
    cell = forged["cells"]["reference"]
    cell["identity"]["verification"] = "local artifacts and runtime"
    for run in cell["runs"]:
        run["metrics"]["choice_accuracy"] = 0.0
    with pytest.raises(ValueError, match="minimum accuracy"):
        compare_reports(deepcopy(forged), forged)


@pytest.mark.parametrize("metric", ["label_flip_rate", "max_probability_drift", "max_score_drift"])
def test_a_saved_pass_is_rechecked_against_its_own_drift_limits(metric):
    baseline = make_report()
    candidate = deepcopy(baseline)
    runs = candidate["cells"]["reference"]["runs"]
    run = runs[-1]
    row = run["observations"][0]
    if metric == "label_flip_rate":
        answer = row["answers"]["route"]["answer"]
        answer.update(choice="deny", probabilities={"allow": 0.49, "deny": 0.51})
    elif metric == "max_probability_drift":
        row["answers"]["route"]["answer"]["probabilities"] = {"allow": 0.53, "deny": 0.47}
    else:
        candidate["limits"]["max_probability_drift"] = 1.0
        row["answers"]["priority"]["answer"].update(
            score=1.01, probabilities={"0": 0.0, "1": 0.99, "2": 0.01})
    raw = {qid: answer["answer"] for qid, answer in row["answers"].items()}
    row["answers"] = qualification._answers({"answers": raw}, QUESTIONS)
    run["comparison"] = qualification._compare(runs[0]["observations"], run["observations"])
    limit_name = "max_label_flip_rate" if metric == "label_flip_rate" else metric
    assert run["comparison"][metric] > candidate["limits"][limit_name]
    with pytest.raises(ValueError, match=metric):
        compare(candidate, baseline)


def test_a_forged_saved_drift_summary_is_recomputed_from_observations():
    baseline = make_report()
    candidate = deepcopy(baseline)
    candidate["cells"]["reference"]["runs"][-1]["comparison"]["max_probability_drift"] = 0.5
    with pytest.raises(ValueError, match="drift summary"):
        compare(candidate, baseline)


@pytest.mark.parametrize("gate", ["missing_accuracy", "accuracy_drop", "p95", "p99", "boundary"])
def test_saved_reports_must_still_satisfy_accuracy_latency_and_boundary_gates(gate):
    baseline = make_report()
    candidate = deepcopy(baseline)
    runs = candidate["cells"]["reference"]["runs"]
    if gate == "missing_accuracy":
        candidate["limits"]["min_accuracy"] = 0.5
        for run in runs:
            run["metrics"] = {name: value for name, value in run["metrics"].items()
                              if not name.endswith("accuracy") and not name.startswith("score_within_")}
    elif gate == "accuracy_drop":
        runs[-1]["metrics"]["choice_accuracy"] = 0.0
    elif gate in ("p95", "p99"):
        candidate["limits"]["max_" + gate + "_ms"] = 0.0
        runs[0]["latency_ms"] = {"p50": 1.0, "p95": 1.0, "p99": 1.0}
    else:
        candidate["limits"]["min_boundary_cases"] = 3
    with pytest.raises(ValueError):
        compare(candidate, baseline)


@pytest.mark.parametrize("defect", ["nonfinite", "negative", "wide_integer", "wrong_count", "forged_quantiles"])
def test_raw_latency_samples_must_be_valid_and_match_saved_percentiles(defect):
    baseline = make_report()
    candidate = deepcopy(baseline)
    run = candidate["cells"]["reference"]["runs"][0]
    if defect == "nonfinite":
        run["latencies_ms"][0] = float("nan")
    elif defect == "negative":
        run["latencies_ms"][0] = -1.0
    elif defect == "wide_integer":
        run["latencies_ms"][0] = 10 ** 400
    elif defect == "wrong_count":
        run["latencies_ms"].pop()
    else:
        run["latencies_ms"] = [1000.0] * len(run["observations"])
        run["latency_ms"] = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        candidate["limits"]["max_p95_ms"] = 10.0
    with pytest.raises(ValueError):
        compare(candidate, baseline)


@pytest.mark.parametrize("field", ["confidence", "answer_confidence", "act_probability"])
def test_uniform_operational_release_drift_is_detected_with_unchanged_primary_probabilities(field):
    baseline = make_report()
    candidate = make_report(operational={field: 0.99})
    assert candidate["passed"] is True
    verdict = compare(candidate, baseline)
    assert verdict["passed"] is False
    assert all(delta["max_probability_drift"] == 0.0 for delta in verdict["cells"]["reference"])
    metric = "max_action_drift" if field == "act_probability" else "max_confidence_drift"
    assert any(metric in failure for failure in verdict["failures"])


def test_saved_normalized_operational_metadata_must_match_the_raw_answer():
    baseline = make_report()
    candidate = deepcopy(baseline)
    observation = candidate["cells"]["reference"]["runs"][0]["observations"][0]["answers"]["route"]
    observation["operational"]["answer_confidence"] = 0.99
    with pytest.raises(ValueError, match="raw answers"):
        compare(candidate, baseline)


@pytest.mark.parametrize("field", ["confidence", "answer_confidence", "act_probability"])
@pytest.mark.parametrize("change", ["missing", "new"])
def test_saved_operational_field_presence_cannot_change_silently(field, change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    removed = candidate if change == "missing" else baseline
    for run in removed["cells"]["reference"]["runs"]:
        for row in run["observations"]:
            answer = row["answers"]["route"]
            answer["answer"].pop("action" if field == "act_probability" else field)
            answer["operational"].pop(field)
    with pytest.raises(ValueError, match="field presence"):
        compare(candidate, baseline)


@pytest.mark.parametrize("field,value", [("low_confidence", True), ("abstention", "abstained")])
def test_uniform_saved_review_flag_changes_are_detected(field, value):
    baseline = make_report()
    candidate = make_report(operational={field: value})
    assert candidate["passed"] is True
    verdict = compare(candidate, baseline)
    assert verdict["passed"] is False
    assert any("operational_flag_flip_rate" in failure for failure in verdict["failures"])


@pytest.mark.parametrize("field", ["low_confidence", "abstention"])
@pytest.mark.parametrize("change", ["missing", "new"])
def test_saved_review_flag_field_presence_cannot_change_silently(field, change):
    baseline = make_report()
    candidate = deepcopy(baseline)
    removed = candidate if change == "missing" else baseline
    for run in removed["cells"]["reference"]["runs"]:
        for row in run["observations"]:
            answer = row["answers"]["route"]
            answer["answer"].pop(field)
            answer["flags"].pop(field)
    with pytest.raises(ValueError, match="field presence"):
        compare(candidate, baseline)


def test_saved_numerical_execution_policy_changes_are_not_comparable():
    baseline = make_report()
    baseline["cells"]["reference"]["identity"]["math_settings"] = {"float32_matmul_precision": "highest"}
    candidate = deepcopy(baseline)
    candidate["cells"]["reference"]["identity"]["math_settings"]["float32_matmul_precision"] = "high"
    verdict = compare(candidate, baseline)
    assert verdict["passed"] is False
    assert any("math_settings" in failure for failure in verdict["failures"])
