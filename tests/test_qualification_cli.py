"""Qualification CLI contracts: strict local inputs, persisted verdicts and exit codes."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from veya import evals_cli, qualification


FIXTURE = Path(__file__).resolve().parents[1] / "research/evals/qualification-v1.json"


def inputs(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    document = {"schema": "veya-qualification-matrix/1", "reference": "reference",
                "cells": [{"name": "reference", "checkpoint": "models/english", "backend": "eager",
                           "device": "cpu", "precision": "fp32", "required": True}],
                "batch_sizes": [1, 2], "repeats": 1,
                "limits": {"min_accuracy": 0.8, "min_boundary_cases": 1}}
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "report.json"
    return corpus, matrix, output, document


def report(passed=True, production_qualified=True):
    return {"schema": "veya-qualification-report/1", "passed": passed,
            "production_qualified": production_qualified,
            "cells": {"reference": {"status": "passed" if passed else "failed"}},
            "failures": [] if passed else ["injected probability drift exceeded the release limit"],
            "coverage": {"completed_cells": ["reference"] if passed else []}}


def command(corpus, matrix, output, *extra):
    return ["qualify", str(corpus), "--matrix", str(matrix), "--json", str(output), *extra]


@pytest.fixture(autouse=True)
def no_model_loading(monkeypatch):
    def forbidden(spec):
        raise AssertionError("CLI contract tests must not load a model")
    monkeypatch.setattr(qualification, "_build_cell", forbidden)


@pytest.mark.parametrize("passed,qualified,exit_code", [(True, True, 0), (False, False, 1),
                                                       (True, False, 1)])
def test_complete_verdict_is_written_for_both_pass_and_failure(monkeypatch, tmp_path, capsys,
                                                              passed, qualified, exit_code):
    corpus, matrix, output, _ = inputs(tmp_path)
    verdict = report(passed, qualified)
    monkeypatch.setattr(qualification, "run_matrix", lambda document, cells: deepcopy(verdict))
    assert evals_cli.main(command(corpus, matrix, output)) == exit_code
    assert json.loads(output.read_text(encoding="utf-8")) == verdict
    captured = capsys.readouterr()
    assert "reference: " + ("passed" if passed else "failed") in captured.out
    if not passed:
        assert "injected probability drift" in captured.err


@pytest.mark.parametrize("removed", ["all", "min_accuracy", "min_boundary_cases"])
def test_release_mode_refuses_missing_quality_gates_before_model_calls(monkeypatch, tmp_path, removed):
    corpus, matrix, output, document = inputs(tmp_path)
    if removed == "all":
        document.pop("limits")
    else:
        document["limits"].pop(removed)
    matrix.write_text(json.dumps(document), encoding="utf-8")
    calls = []
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: calls.append(args))
    assert evals_cli.main(command(corpus, matrix, output)) == 2
    assert calls == []
    assert not output.exists()


def test_diagnostic_agreement_run_can_pass_without_claiming_production(monkeypatch, tmp_path):
    corpus, matrix, output, document = inputs(tmp_path)
    document.pop("limits")
    matrix.write_text(json.dumps(document), encoding="utf-8")
    verdict = report(production_qualified=False)
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: deepcopy(verdict))
    assert evals_cli.main(command(corpus, matrix, output, "--diagnostic")) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["production_qualified"] is False


def test_diagnostic_flag_does_not_turn_failed_measurements_into_success(monkeypatch, tmp_path):
    corpus, matrix, output, document = inputs(tmp_path)
    document.pop("limits")
    matrix.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: report(False, False))
    assert evals_cli.main(command(corpus, matrix, output, "--diagnostic")) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False


def test_paths_resolve_relative_to_matrix_file_and_keep_corpus_independent(monkeypatch, tmp_path):
    corpus, matrix, output, document = inputs(tmp_path)
    document["cells"].append({"name": "onnx", "checkpoint": "models/english", "backend": "onnx",
                              "device": "cpu", "precision": "graph", "required": False,
                              "onnx_path": "graphs/english.onnx"})
    matrix.write_text(json.dumps(document), encoding="utf-8")
    seen = []

    def run(document, cells):
        seen.append((document, cells))
        return report()

    monkeypatch.setattr(qualification, "run_matrix", run)
    assert evals_cli.main(command(corpus, matrix, output)) == 0
    loaded, matrix_document = seen[0]
    assert len(loaded["cases"]) == 22
    assert matrix_document["cells"][0]["checkpoint"] == str((tmp_path / "models/english").resolve())
    assert matrix_document["cells"][1]["onnx_path"] == str((tmp_path / "graphs/english.onnx").resolve())
    assert json.loads(matrix.read_text(encoding="utf-8")) == document


def test_validation_reports_case_and_source_family_counts_without_model_calls(tmp_path, capsys):
    corpus, _, _, _ = inputs(tmp_path)
    assert evals_cli.main(["qualify-validate", str(corpus)]) == 0
    output = capsys.readouterr().out
    assert "22 cases" in output
    assert "5 source families" in output
    assert "split leakage checks passed" in output


@pytest.mark.parametrize("location", ["corpus", "matrix"])
@pytest.mark.parametrize("defect", ["duplicate_keys", "nonfinite", "invalid_json", "missing_file"])
def test_malformed_or_unreadable_documents_are_usage_errors(tmp_path, location, defect):
    corpus, matrix, output, _ = inputs(tmp_path)
    path = corpus if location == "corpus" else matrix
    if defect == "duplicate_keys":
        # Duplicate values must be refused even when either value would be legal alone.
        path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    elif defect == "nonfinite":
        path.write_text('{"number":NaN}', encoding="utf-8")
    elif defect == "invalid_json":
        path.write_text("{", encoding="utf-8")
    else:
        path.unlink()
    assert evals_cli.main(command(corpus, matrix, output)) == 2
    assert not output.exists()


@pytest.mark.parametrize("defect", ["bad_gold", "family_leak", "unknown_backend", "unknown_threshold"])
def test_invalid_release_contracts_are_usage_errors_before_execution(monkeypatch, tmp_path, defect):
    corpus, matrix, output, document = inputs(tmp_path)
    if defect in ("bad_gold", "family_leak"):
        corpus_document = json.loads(corpus.read_text(encoding="utf-8"))
        if defect == "bad_gold":
            corpus_document["cases"][0]["expected"]["department"] = "nonexistent-option"
        else:
            corpus_document["cases"][1]["split"] = "training"
        corpus.write_text(json.dumps(corpus_document), encoding="utf-8")
    elif defect == "unknown_backend":
        document["cells"][0]["backend"] = "automatic-fallback"
        matrix.write_text(json.dumps(document), encoding="utf-8")
    else:
        document["limits"]["misspelled_limit"] = 0.0
        matrix.write_text(json.dumps(document), encoding="utf-8")
    calls = []
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: calls.append(args))
    assert evals_cli.main(command(corpus, matrix, output)) == 2
    assert calls == []
    assert not output.exists()


def test_invalid_gold_is_also_refused_by_validation_command(tmp_path):
    corpus, _, _, _ = inputs(tmp_path)
    document = json.loads(corpus.read_text(encoding="utf-8"))
    document["cases"][0]["expected"]["department"] = "nonexistent-option"
    corpus.write_text(json.dumps(document), encoding="utf-8")
    assert evals_cli.main(["qualify-validate", str(corpus)]) == 2


def test_report_write_failure_has_a_distinct_usage_error(monkeypatch, tmp_path):
    corpus, matrix, _, _ = inputs(tmp_path)
    output = tmp_path / "missing-directory" / "report.json"
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: report())
    assert evals_cli.main(command(corpus, matrix, output)) == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", [qualification._Unavailable, ValueError])
def test_required_cell_load_failure_writes_a_failed_release_verdict(monkeypatch, tmp_path, failure):
    corpus, matrix, output, _ = inputs(tmp_path)

    def unavailable(spec):
        raise failure("injected required-cell load failure")

    monkeypatch.setattr(qualification, "_build_cell", unavailable)
    assert evals_cli.main(command(corpus, matrix, output)) == 1
    verdict = json.loads(output.read_text(encoding="utf-8"))
    assert verdict["passed"] is False
    assert verdict["production_qualified"] is False
    assert verdict["cells"]["reference"]["status"] == "failed"
    assert any("injected required-cell load failure" in failure for failure in verdict["failures"])


def test_wide_integer_limits_are_usage_errors_before_model_calls(monkeypatch, tmp_path):
    corpus, matrix, output, document = inputs(tmp_path)
    document["limits"]["max_probability_drift"] = 10 ** 400
    matrix.write_text(json.dumps(document), encoding="utf-8")
    calls = []
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: calls.append(args))
    assert evals_cli.main(command(corpus, matrix, output)) == 2
    assert calls == []
    assert not output.exists()


def saved_report(**kwargs):
    helper = Path(__file__).with_name("test_qualification_baseline.py")
    return runpy.run_path(str(helper))["make_report"](**kwargs)


def comparison_inputs(tmp_path, *, flip=False, operational=None):
    baseline = tmp_path / "reviewed-baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
    candidate.write_text(json.dumps(saved_report(flip=flip, operational=operational)), encoding="utf-8")
    return candidate, baseline, output


def comparison_command(candidate, baseline, output, *extra):
    return ["qualify-compare", str(candidate), "--baseline", str(baseline), "--json", str(output), *extra]


@pytest.mark.parametrize("flip,exit_code", [(False, 0), (True, 1)])
def test_offline_saved_release_comparison_persists_pass_and_global_flip_failures(tmp_path, flip, exit_code):
    candidate, baseline, output = comparison_inputs(tmp_path, flip=flip)
    assert evals_cli.main(comparison_command(candidate, baseline, output, "--diagnostic")) == exit_code
    verdict = json.loads(output.read_text(encoding="utf-8"))
    assert verdict["schema"] == "veya-qualification-comparison/1"
    assert verdict["passed"] is (not flip)
    assert bool(verdict["failures"]) is flip


def test_offline_source_revision_changes_can_pass_when_decisions_are_unchanged(tmp_path):
    candidate, baseline, output = comparison_inputs(tmp_path)
    document = json.loads(candidate.read_text(encoding="utf-8"))
    document["source_sha256"]["qualification.py"] = "b" * 64
    candidate.write_text(json.dumps(document), encoding="utf-8")
    assert evals_cli.main(comparison_command(candidate, baseline, output, "--diagnostic")) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


@pytest.mark.parametrize("defect", ["raw_tamper", "missing_candidate", "duplicate_baseline_keys", "failed_baseline"])
def test_invalid_offline_saved_reports_are_usage_errors(tmp_path, defect):
    candidate, baseline, output = comparison_inputs(tmp_path)
    if defect == "raw_tamper":
        document = json.loads(candidate.read_text(encoding="utf-8"))
        document["cells"]["reference"]["runs"][0]["observations"][0]["answers"]["route"]["answer"]["choice"] = "deny"
        candidate.write_text(json.dumps(document), encoding="utf-8")
    elif defect == "missing_candidate":
        candidate.unlink()
    elif defect == "duplicate_baseline_keys":
        baseline.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    else:
        document = json.loads(baseline.read_text(encoding="utf-8"))
        document["passed"] = False
        document["failures"] = ["already failed"]
        baseline.write_text(json.dumps(document), encoding="utf-8")
    assert evals_cli.main(comparison_command(candidate, baseline, output, "--diagnostic")) == 2
    assert not output.exists()


def test_offline_comparison_requires_diagnostic_opt_in_for_unverified_reports(tmp_path):
    candidate, baseline, output = comparison_inputs(tmp_path)
    assert evals_cli.main(comparison_command(candidate, baseline, output)) == 2
    assert not output.exists()


def test_live_release_rejects_unverified_baseline_before_loading_target_models(monkeypatch, tmp_path):
    corpus, matrix, output, _ = inputs(tmp_path)
    baseline = tmp_path / "reviewed-baseline.json"
    baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
    calls = []
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: calls.append(args))
    assert evals_cli.main(command(corpus, matrix, output, "--baseline", str(baseline))) == 2
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("flip,exit_code", [(False, 0), (True, 1)])
def test_live_diagnostic_comparison_uses_reviewed_prior_release(monkeypatch, tmp_path, flip, exit_code):
    corpus, matrix, output, _ = inputs(tmp_path)
    baseline = tmp_path / "reviewed-baseline.json"
    baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
    candidate = saved_report(flip=flip)
    assert candidate["passed"] is True
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: deepcopy(candidate))
    args = command(corpus, matrix, output, "--baseline", str(baseline), "--diagnostic")
    assert evals_cli.main(args) == exit_code
    verdict = json.loads(output.read_text(encoding="utf-8"))
    assert verdict["baseline_comparison"]["passed"] is (not flip)
    assert verdict["passed"] is (not flip)
    assert verdict["production_qualified"] is False
    if flip:
        assert any("saved release baseline" in failure for failure in verdict["failures"])


def test_an_already_failed_live_run_is_saved_without_masking_its_error_with_a_comparison(monkeypatch, tmp_path):
    corpus, matrix, output, _ = inputs(tmp_path)
    baseline = tmp_path / "reviewed-baseline.json"
    baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: report(False, False))

    def forbidden(*args, **kwargs):
        raise AssertionError("a failed run must preserve its own failure")

    monkeypatch.setattr(qualification, "compare_reports", forbidden)
    args = command(corpus, matrix, output, "--baseline", str(baseline), "--diagnostic")
    assert evals_cli.main(args) == 1
    verdict = json.loads(output.read_text(encoding="utf-8"))
    assert verdict["passed"] is False
    assert "baseline_comparison" not in verdict
    assert "injected probability drift" in verdict["failures"][0]


@pytest.mark.parametrize("location", ["candidate", "baseline"])
@pytest.mark.parametrize("defect", ["nonfinite", "negative", "wide_integer", "wrong_count", "forged_quantiles"])
def test_offline_comparison_refuses_invalid_raw_latency_evidence(tmp_path, location, defect):
    candidate, baseline, output = comparison_inputs(tmp_path)
    path = candidate if location == "candidate" else baseline
    document = json.loads(path.read_text(encoding="utf-8"))
    run = document["cells"]["reference"]["runs"][0]
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
        document["limits"]["max_p95_ms"] = 10.0
    path.write_text(json.dumps(document), encoding="utf-8")
    assert evals_cli.main(comparison_command(candidate, baseline, output, "--diagnostic")) == 2
    assert not output.exists()


@pytest.mark.parametrize("field", ["answer_confidence", "act_probability"])
def test_offline_release_comparison_detects_operational_drift_with_stable_primary_answers(tmp_path, field):
    candidate, baseline, output = comparison_inputs(tmp_path, operational={field: 0.99})
    assert evals_cli.main(comparison_command(candidate, baseline, output, "--diagnostic")) == 1
    verdict = json.loads(output.read_text(encoding="utf-8"))
    assert verdict["passed"] is False
    assert all(delta["max_probability_drift"] == 0.0 for delta in verdict["cells"]["reference"])
