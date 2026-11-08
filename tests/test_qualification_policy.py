"""Opt-in decision-policy identity and calibration-status qualification contracts."""
from copy import deepcopy
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

import veya.qualification as qualification
from veya.decision_policy import DECISION_POLICIES, resolve_decision_policy


ROOT = Path(__file__).resolve().parents[1]
CORE = runpy.run_path(str(Path(__file__).with_name("test_qualification.py")))
BASELINE = runpy.run_path(str(Path(__file__).with_name("test_qualification_baseline.py")))
FixtureRunner = CORE["FixtureRunner"]


@pytest.mark.parametrize("value", [None, "unknown", "", True, 1, {}, []])
def test_invalid_matrix_decision_policy_is_refused_before_loading(monkeypatch, value):
    matrix = CORE["matrix"]()
    matrix["cells"][0]["decision_policy"] = value
    loaded = []
    monkeypatch.setattr(qualification, "_build_cell", lambda spec: loaded.append(spec))
    with pytest.raises(ValueError, match="decision_policy"):
        qualification.run_matrix(CORE["corpus"](), matrix)
    assert loaded == []


@pytest.mark.parametrize("policy", DECISION_POLICIES)
def test_supported_matrix_policy_is_preserved_without_mutating_input(policy):
    matrix = CORE["matrix"]()
    matrix["cells"][0]["decision_policy"] = policy
    original = deepcopy(matrix)
    assert qualification.validate_matrix(matrix) == original
    assert matrix == original


def install_fake_cells(monkeypatch, *, checkpoint_default="canonical", ignore_explicit=False):
    torch = pytest.importorskip("torch")
    import veya.agent as agent_module
    import veya.onnx_agent as onnx_module
    import veya.consistency as consistency

    created = []
    omitted = object()

    class TorchAgent:
        hooks = ()
        backend = "eager"
        dtype = torch.float32
        amp_enabled = False

        def __init__(self, checkpoint, *, decision_policy=omitted, **kwargs):
            self.device = torch.device("cpu")
            self.model = torch.nn.Identity()
            self.cfg = {"decision_policy": checkpoint_default}
            self.decision_policy = resolve_decision_policy(
                None if decision_policy is omitted or ignore_explicit else decision_policy, self.cfg)
            self.closed = False
            created.append({"kind": "eager", "agent": self,
                            "explicit": decision_policy is not omitted, "value": decision_policy})

        def close(self):
            self.closed = True
            self.model = None

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_session_options(self):
            return SimpleNamespace(intra_op_num_threads=1, inter_op_num_threads=1)

    class OnnxAgent:
        hooks = ()

        def __init__(self, checkpoint, *, decision_policy=omitted, **kwargs):
            self.session = Session()
            self.cfg = {"decision_policy": checkpoint_default}
            self.decision_policy = resolve_decision_policy(
                None if decision_policy is omitted or ignore_explicit else decision_policy, self.cfg)
            self.closed = False
            created.append({"kind": "onnx", "agent": self,
                            "explicit": decision_policy is not omitted, "value": decision_policy})

        def close(self):
            self.closed = True
            self.session = None

    monkeypatch.setattr(agent_module, "Agent", TorchAgent)
    monkeypatch.setattr(onnx_module, "_NativeONNXAgent", OnnxAgent)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="fixture-runtime"))
    monkeypatch.setattr(qualification, "_artifact_identity", lambda *args: {"weights": "a" * 64})
    monkeypatch.setattr(consistency, "_fingerprint", lambda agent: ["fixture", agent.decision_policy])
    return created


def cell(backend, *, policy=None, explicit=False):
    spec = {"name": "fixture", "checkpoint": "/local/checkpoint", "backend": backend,
            "device": "cpu", "precision": "graph" if backend == "onnx" else "fp32"}
    if backend == "onnx":
        spec["onnx_path"] = "/local/checkpoint.onnx"
    if explicit:
        spec["decision_policy"] = policy
    return spec


@pytest.mark.parametrize("backend", ["eager", "onnx"])
@pytest.mark.parametrize("policy", DECISION_POLICIES)
def test_factory_forwards_explicit_policy_to_both_constructor_paths(monkeypatch, backend, policy):
    created = install_fake_cells(monkeypatch)
    runner = qualification._build_cell(cell(backend, policy=policy, explicit=True))
    agent = runner.agent
    try:
        assert created[0]["explicit"] is True
        assert created[0]["value"] == policy
        assert agent.decision_policy == policy
        if policy == "canonical":
            assert "decision_policy" not in runner.identity
        else:
            assert runner.identity["decision_policy"] == policy
    finally:
        runner.close()
    assert agent.closed


@pytest.mark.parametrize("backend", ["eager", "onnx"])
def test_factory_refuses_runtime_that_ignores_requested_policy_and_closes_it(monkeypatch, backend):
    created = install_fake_cells(monkeypatch, ignore_explicit=True)
    with pytest.raises(ValueError, match="did not activate the requested decision policy"):
        qualification._build_cell(cell(backend, policy="context-balanced-v1", explicit=True))
    assert created[0]["explicit"] is True
    assert created[0]["agent"].closed


@pytest.mark.parametrize("backend", ["eager", "onnx"])
@pytest.mark.parametrize("default", DECISION_POLICIES)
def test_omitted_policy_uses_checkpoint_default_without_adding_a_constructor_keyword(monkeypatch, backend, default):
    created = install_fake_cells(monkeypatch, checkpoint_default=default)
    runner = qualification._build_cell(cell(backend))
    try:
        assert created[0]["explicit"] is False
        assert runner.agent.decision_policy == default
        assert runner.identity.get("decision_policy", "canonical") == default
        if default == "canonical":
            assert "decision_policy" not in runner.identity
    finally:
        runner.close()


@pytest.mark.parametrize("backend", ["eager", "onnx"])
def test_explicit_canonical_policy_can_override_a_noncanonical_checkpoint_default(monkeypatch, backend):
    created = install_fake_cells(monkeypatch, checkpoint_default="context-balanced-v1")
    runner = qualification._build_cell(cell(backend, policy="canonical", explicit=True))
    try:
        assert created[0]["explicit"] is True
        assert created[0]["value"] == "canonical"
        assert runner.agent.decision_policy == "canonical"
        assert "decision_policy" not in runner.identity
    finally:
        runner.close()


@pytest.mark.parametrize("policy", DECISION_POLICIES)
def test_runtime_policy_value_cannot_change_even_if_the_fingerprint_is_held_constant(monkeypatch, policy):
    import veya.consistency as consistency

    install_fake_cells(monkeypatch)
    runner = qualification._build_cell(cell("eager", policy=policy, explicit=True))
    original = deepcopy(runner.identity["model_fingerprint"])
    monkeypatch.setattr(consistency, "_fingerprint", lambda agent: deepcopy(original))
    runner.agent.decision_policy = "context-balanced-v1" if policy == "canonical" else "canonical"
    try:
        with pytest.raises(ValueError, match="policy"):
            runner._check()
    finally:
        runner.close()


def test_runtime_policy_fingerprint_mutation_is_also_refused(monkeypatch):
    import veya.consistency as consistency

    install_fake_cells(monkeypatch)
    runner = qualification._build_cell(cell("eager", policy="context-balanced-v1", explicit=True))
    monkeypatch.setattr(consistency, "_fingerprint", lambda agent: ["changed-policy-fingerprint"])
    try:
        with pytest.raises(ValueError, match="configuration|policy"):
            runner._check()
    finally:
        runner.close()


def test_saved_release_policy_identity_changes_are_not_comparable():
    baseline = BASELINE["make_report"]()
    candidate = deepcopy(baseline)
    candidate["cells"]["reference"]["identity"]["decision_policy"] = "context-balanced-v1"
    verdict = qualification.compare_reports(candidate, baseline, allow_unverified_identity=True)
    assert verdict["passed"] is False
    assert any("decision_policy" in failure for failure in verdict["failures"])


@pytest.mark.parametrize("policy", [None, "unknown", "", True, 1, {}])
def test_invalid_saved_policy_identity_is_refused_even_when_both_reports_match(policy):
    report = BASELINE["make_report"]()
    report["cells"]["reference"]["identity"]["decision_policy"] = policy
    with pytest.raises(ValueError, match="decision.policy"):
        qualification.compare_reports(deepcopy(report), report, allow_unverified_identity=True)


class CalibrationStatusRunner(FixtureRunner):
    def __init__(self, value=False, *, present=True):
        self.value, self.present = value, present

    def predict(self, state, questions, model=None):
        result = super().predict(state, questions, model=model)
        if self.present:
            for answer in result["answers"].values():
                answer["confidence_calibrated"] = self.value
        return result


def diagnostic(runner):
    return qualification.qualify(CORE["corpus"](), {"fixture": runner}, reference="fixture",
                                 batch_sizes=(1, 2), repeats=1, allow_unverified_identity=True)


def test_absent_canonical_calibration_flags_keep_the_existing_normalized_contract():
    report = diagnostic(FixtureRunner())
    assert report["passed"] is True
    for run in report["cells"]["fixture"]["runs"]:
        for row in run["observations"]:
            assert all("confidence_calibrated" not in answer["flags"] for answer in row["answers"].values())


@pytest.mark.parametrize("value", [False, True])
def test_boolean_calibration_status_is_preserved_in_every_observation(value):
    report = diagnostic(CalibrationStatusRunner(value))
    assert report["passed"] is True
    for run in report["cells"]["fixture"]["runs"]:
        for row in run["observations"]:
            assert all(answer["flags"]["confidence_calibrated"] is value for answer in row["answers"].values())


@pytest.mark.parametrize("value", [0, 1, "false", None, [], float("nan")])
def test_confidence_calibrated_flag_requires_a_boolean(value):
    report = diagnostic(CalibrationStatusRunner(value))
    assert report["passed"] is False
    assert report["cells"]["fixture"]["status"] == "failed"


def test_changed_calibration_status_is_an_operational_flag_flip():
    class Changed(CalibrationStatusRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            results[0]["answers"]["route"]["confidence_calibrated"] = True
            return results

    report = diagnostic(Changed())
    assert report["passed"] is False
    assert any("operational_flag_flip_rate" in failure for failure in report["failures"])


@pytest.mark.parametrize("change", ["missing", "new"])
def test_calibration_status_field_presence_is_part_of_the_live_contract(change):
    class Changed(CalibrationStatusRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            if change == "missing":
                results[0]["answers"]["route"].pop("confidence_calibrated")
            else:
                results[0]["answers"]["route"]["confidence_calibrated"] = False
            return results

    report = diagnostic(Changed(present=change != "new"))
    assert report["passed"] is False
    assert any("field presence" in failure for failure in report["failures"])


def test_uniform_saved_calibration_status_change_is_detected_across_releases():
    baseline = diagnostic(CalibrationStatusRunner(False))
    candidate = diagnostic(CalibrationStatusRunner(True))
    assert baseline["passed"] is True and candidate["passed"] is True
    verdict = qualification.compare_reports(candidate, baseline, allow_unverified_identity=True)
    assert verdict["passed"] is False
    assert any("operational_flag_flip_rate" in failure for failure in verdict["failures"])


@pytest.mark.parametrize("change", ["missing", "new"])
def test_saved_calibration_status_field_presence_cannot_change_silently(change):
    baseline = diagnostic(CalibrationStatusRunner(present=change == "missing"))
    candidate = diagnostic(CalibrationStatusRunner(present=change != "missing"))
    with pytest.raises(ValueError, match="field presence"):
        qualification.compare_reports(candidate, baseline, allow_unverified_identity=True)


@pytest.mark.parametrize("policy", [None, "canonical", "context-balanced-v1"])
def test_local_qualification_script_only_adds_policy_when_explicit(monkeypatch, tmp_path, policy):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    namespace = runpy.run_path(str(ROOT / "research/scripts/qualify_local.py"))
    out = tmp_path / "policy-qualification.json"
    argv = ["qualify_local", "--checkpoint", str(tmp_path / "checkpoint"),
            "--onnx", str(tmp_path / "checkpoint.onnx"), "--out", str(out), "--diagnostic"]
    if policy is not None:
        argv.extend(["--decision-policy", policy])
    monkeypatch.setattr(sys, "argv", argv)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VEYA_THREADS"):
        monkeypatch.setenv(name, "fixture-before")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda count: None, set_num_interop_threads=lambda count: None,
        get_num_threads=lambda: 4, get_num_interop_threads=lambda: 1))
    captured = []

    def run(corpus, matrix):
        captured.append(deepcopy(matrix))
        return {"passed": True, "production_qualified": False, "failures": [], "coverage": {"boundary_cases": 0},
                "cells": {spec["name"]: {"status": "passed", "runs": []} for spec in matrix["cells"]}}

    monkeypatch.setattr(qualification, "run_matrix", run)
    assert namespace["main"]() == 0
    assert len(captured[0]["cells"]) == 2
    for spec in captured[0]["cells"]:
        if policy is None:
            assert "decision_policy" not in spec
        else:
            assert spec["decision_policy"] == policy
    assert out.is_file()


def test_local_script_policy_help_and_invalid_choice_stop_before_loading(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    namespace = runpy.run_path(str(ROOT / "research/scripts/qualify_local.py"))
    monkeypatch.setattr(sys, "argv", ["qualify_local", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        namespace["main"]()
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--decision-policy" in help_text
    assert all(policy in help_text for policy in DECISION_POLICIES)
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(qualification, "run_matrix", lambda *args: pytest.fail("invalid policy reached inference"))
    monkeypatch.setattr(sys, "argv", ["qualify_local", "--checkpoint", str(tmp_path / "unused"), "--out", str(output),
                                     "--decision-policy", "unknown"])
    with pytest.raises(SystemExit) as invalid:
        namespace["main"]()
    assert invalid.value.code == 2
    assert not output.exists()
