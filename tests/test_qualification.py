"""Release-qualification failure controls; no model weights or hosted services."""
from copy import deepcopy
from contextlib import nullcontext
import builtins
import gc
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest

import laya.qualification as qualification
from laya.qualification import qualify, run_matrix, split_families, validate_corpus, validate_matrix


def test_evidence_training_rejects_conversation_state():
    from research.scripts.train_evidence_repair import check_splits

    row = {"family": "long-conversation", "state": [{"role": "user", "content": "Latest update"}],
           "questions": {"decision": {"type": "noul"}}}
    with pytest.raises(ValueError, match="string or object states"):
        check_splits({"training": [row]})


def test_evidence_verification_allows_encoder_only_when_requested(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from research.scripts.verify_checkpoint_repair import frozen_weights

    base, candidate = tmp_path / "base", tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    original = {"encoder.weight": torch.zeros(2), "act_head.weight": torch.zeros(1)}
    adapted = {**original, "encoder.weight": torch.ones(2)}
    safetensors.save_file(original, base / "model.safetensors")
    safetensors.save_file(adapted, candidate / "model.safetensors")
    with pytest.raises(AssertionError, match="encoder.weight"):
        frozen_weights(base, candidate)
    assert frozen_weights(base, candidate, allow_encoder_update=True)["updated_tensors"] == ["encoder.weight"]
    safetensors.save_file({**adapted, "act_head.weight": torch.ones(1)}, candidate / "model.safetensors")
    with pytest.raises(AssertionError, match="act_head.weight"):
        frozen_weights(base, candidate, allow_encoder_update=True)


def test_level_zero_repair_loss_uses_semantic_levels_after_rotation():
    torch = pytest.importorskip("torch")
    from research.scripts.train_checkpoint_repair import level_zero_loss

    logits = torch.tensor([[-1., 0., 2.], [-1., 2., -2.]], requires_grad=True)
    gold = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    anchor = torch.tensor([[.1, .2, .7], [.15, .72, .13]])
    levels = torch.arange(3).expand(2, 3)
    kinds = torch.tensor([1, 1])
    temperatures = torch.tensor([.85, .94, 1.65])
    loss = level_zero_loss(logits, gold, anchor, levels, kinds, temperatures)
    order = [2, 0, 1]
    rotated = level_zero_loss(logits[:, order], gold[:, order], anchor[:, order],
                              levels[:, order], kinds, temperatures)
    torch.testing.assert_close(loss, rotated)
    loss.backward()
    assert logits.grad[0, 0] < 0 < logits.grad[0, 2]


def test_artifact_identity_detects_changed_sparse_external_data(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    (tmp_path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "values.data"
    sidecar.write_bytes(b"\x00\x00\x80\x3f")
    values = helper.make_tensor("sparse", TensorProto.FLOAT, [1], [1.0])
    values.ClearField("float_data")
    values.data_location = TensorProto.EXTERNAL
    values.external_data.add(key="location", value=sidecar.name)
    indices = helper.make_tensor("indices", TensorProto.INT64, [1], [0])
    graph = helper.make_graph([], "sparse", [], [])
    graph.sparse_initializer.append(helper.make_sparse_tensor(values, indices, [2]))
    path = tmp_path / "model.onnx"
    onnx.save_model(helper.make_model(graph), path)
    before = qualification._artifact_identity(tmp_path, path)
    sidecar.write_bytes(b"\x00\x00\x00\x40")
    after = qualification._artifact_identity(tmp_path, path)
    assert after["onnx/model.onnx"] == before["onnx/model.onnx"]
    assert after["onnx-external/values.data"] != before["onnx-external/values.data"]


def test_onnx_artifact_scan_accepts_legacy_repeated_descriptors(tmp_path, monkeypatch):
    (tmp_path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "weights.data").write_bytes(b"weights")
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"graph")
    tensor = SimpleNamespace(DESCRIPTOR=SimpleNamespace(full_name="onnx.TensorProto"),
                             external_data=[SimpleNamespace(key="location", value="weights.data")],
                             ListFields=lambda: [])
    legacy = SimpleNamespace(message_type=object(), label=3, LABEL_REPEATED=3)
    model = SimpleNamespace(DESCRIPTOR=SimpleNamespace(full_name="onnx.ModelProto"),
                            ListFields=lambda: [(legacy, [tensor])])
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=lambda *a, **k: model))
    identity = qualification._artifact_identity(tmp_path, graph)
    assert "onnx-external/weights.data" in identity


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Allow or deny?",
              "criteria": {"allow": "Authorized requests", "deny": "Unauthorized requests"}},
    "authorized": {"type": "noul", "instructions": "Is this authorized?"},
    "priority": {"type": "score", "instructions": "Rate the priority.",
                 "criteria": ["Low", "Medium", "High"]},
}


def corpus():
    rows = []
    for keep in (True, False):
        family = "allow-family" if keep else "deny-family"
        for variant in ("base", "paraphrase"):
            rows.append({"id": family + "-" + variant, "family": family,
                         "split": "evaluation", "variant": variant,
                         "state": {"case": family + "-" + variant, "keep": keep},
                         "questions": deepcopy(QUESTIONS),
                         "expected": {"route": "allow" if keep else "deny",
                                      "authorized": keep, "priority": 1},
                         "tags": ["boundary-candidate"]})
    return {"schema": "veya-qualification-corpus/1", "id": "qualification-test-v1", "cases": rows}


class FixtureRunner:
    """An explicitly unverified runner, so fixture success can never qualify production."""

    def predict(self, state, questions, model=None):
        keep = bool(state.get("keep", True)) if isinstance(state, dict) else True
        probability = 0.51 if keep else 0.49
        return {"answers": {
            "route": {"type": "choice", "choice": "allow" if keep else "deny",
                      "probabilities": {"allow": probability, "deny": 1 - probability},
                      "confidence": 0.01, "answer_confidence": 0.51},
            "authorized": {"type": "noul", "noul": probability,
                           "confidence": 0.51, "answer_confidence": 0.51},
            "priority": {"type": "score", "score": 1.0,
                         "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
                         "legend": {"0": "Low", "1": "Medium", "2": "High"},
                         "confidence": 1.0, "answer_confidence": 1.0}}, "usage": {}}

    def predict_batch(self, states, questions, model=None, batch_size=None):
        return [self.predict(state, questions, model=model) for state in states]


def diagnostic(runner=None, document=None, **kwargs):
    return qualify(document or corpus(), {"fixture": runner or FixtureRunner()},
                   reference="fixture", allow_unverified_identity=True, **kwargs)


def test_corpus_validation_returns_an_independent_document():
    original = corpus()
    validated = validate_corpus(original)
    assert validated == original
    validated["cases"][0]["state"]["keep"] = False
    assert original["cases"][0]["state"]["keep"] is True


@pytest.mark.parametrize("change", ["family_leak", "input_leak", "duplicate_id", "unknown_gold",
                                   "invalid_choice", "invalid_noul", "invalid_score", "nonfinite_state",
                                   "nonfinite_gold", "unknown_split", "empty_cases", "unknown_schema"])
def test_invalid_or_leaking_corpora_are_refused(change):
    document = corpus()
    row = document["cases"][0]
    if change == "family_leak":
        document["cases"][1]["split"] = "training"
    elif change == "input_leak":
        duplicate = deepcopy(row)
        duplicate.update(id="elsewhere", family="another-family", split="training")
        document["cases"].append(duplicate)
    elif change == "duplicate_id":
        document["cases"][1]["id"] = row["id"]
    elif change == "unknown_gold":
        row["expected"]["missing"] = True
    elif change == "invalid_choice":
        row["expected"]["route"] = "unknown-option"
    elif change == "invalid_noul":
        row["expected"]["authorized"] = 1
    elif change == "invalid_score":
        row["expected"]["priority"] = 7
    elif change == "nonfinite_state":
        row["state"]["unexpected"] = float("nan")
    elif change == "nonfinite_gold":
        row["expected"]["priority"] = float("inf")
    elif change == "unknown_split":
        row["split"] = "maybe"
    elif change == "empty_cases":
        document["cases"] = []
    elif change == "unknown_schema":
        document["schema"] = "veya-qualification-corpus/999"
    with pytest.raises(ValueError):
        validate_corpus(document)


def test_family_splitting_is_reproducible_and_never_splits_a_family():
    document = corpus()
    for i in range(8):
        row = deepcopy(document["cases"][0])
        row.update(id="extra-%d" % i, family="extra-family-%d" % i)
        row["state"]["case"] = row["id"]
        document["cases"].append(row)
    first = split_families(document, evaluation_fraction=0.2, calibration_fraction=0.2, seed=17)
    assert first == split_families(document, evaluation_fraction=0.2, calibration_fraction=0.2, seed=17)
    assert document["cases"][0]["split"] == "evaluation"
    family_splits = {}
    for row in first["cases"]:
        family_splits.setdefault(row["family"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert {row["split"] for row in first["cases"]} == {"training", "calibration", "evaluation"}


@pytest.mark.parametrize("fractions", [(float("nan"), 0.2), (0.2, float("inf")),
                                      (-0.1, 0.2), (0.8, 0.8)])
def test_invalid_family_split_fractions_are_refused(fractions):
    with pytest.raises(ValueError):
        split_families(corpus(), evaluation_fraction=fractions[0], calibration_fraction=fractions[1])


def test_unknown_runtime_identity_cannot_pass_a_production_gate():
    with pytest.raises(ValueError):
        qualify(corpus(), {"fixture": FixtureRunner()}, reference="fixture")


def test_perfect_fixture_is_only_a_diagnostic_success_and_records_matrix():
    report = diagnostic(batch_sizes=(1, 2), repeats=2)
    assert report["schema"] == "veya-qualification-report/1"
    assert report["passed"] is True
    assert report["production_qualified"] is False
    assert report["failures"] == []
    cell = report["cells"]["fixture"]
    assert cell["status"] == "passed"
    assert {run["batch_size"] for run in cell["runs"]} == {1, 2}
    for run in cell["runs"]:
        assert run["comparison"]["label_flip_rate"] == 0.0
        assert run["comparison"]["max_probability_drift"] == 0.0
        assert run["comparison"]["max_score_drift"] == 0.0
        assert set(run["latency_ms"]) >= {"p50", "p95", "p99"}
        assert {row["id"] for row in run["observations"]} == {row["id"] for row in corpus()["cases"]}


class BatchDrift(FixtureRunner):
    def __init__(self, kind):
        self.kind = kind

    def predict_batch(self, states, questions, model=None, batch_size=None):
        results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
        for result in results:
            answers = result["answers"]
            if self.kind == "label":
                answer = answers["route"]
                answer["choice"] = "deny" if answer["choice"] == "allow" else "allow"
                answer["probabilities"] = {key: 1 - value for key, value in answer["probabilities"].items()}
            elif self.kind == "probability":
                answer = answers["route"]
                direction = 0.01 if answer["choice"] == "allow" else -0.01
                answer["probabilities"]["allow"] += direction
                answer["probabilities"]["deny"] -= direction
            else:
                answers["priority"].update(score=1.01, probabilities={"0": 0.0, "1": 0.99, "2": 0.01})
        return results


@pytest.mark.parametrize("kind", ["label", "probability", "score"])
def test_injected_batch_drift_fails_the_gate(kind):
    limits = {"max_probability_drift": 1.0} if kind == "score" else None
    report = diagnostic(BatchDrift(kind), batch_sizes=(1, 2), repeats=1, limits=limits)
    assert report["passed"] is False
    assert report["cells"]["fixture"]["status"] == "failed"
    assert report["failures"]


def test_unrelated_batch_neighbour_cannot_change_original_decisions():
    known = {json.dumps(row["state"], sort_keys=True) for row in corpus()["cases"]}

    class NeighbourSensitive(FixtureRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            if any(json.dumps(state, sort_keys=True) not in known for state in states):
                for state, result in zip(states, results):
                    if json.dumps(state, sort_keys=True) in known:
                        answer = result["answers"]["authorized"]
                        answer["noul"] = 1 - answer["noul"]
            return results

    report = diagnostic(NeighbourSensitive(), batch_sizes=(1, 2), repeats=1)
    assert report["passed"] is False
    assert report["failures"]


@pytest.mark.parametrize("defect", ["short_batch", "long_batch", "missing_answer", "nonfinite", "raises"])
def test_runtime_errors_and_incomplete_results_fail_instead_of_skipping(defect):
    class Broken(FixtureRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            if defect == "raises":
                raise RuntimeError("injected runner failure")
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            if defect == "short_batch":
                return results[:-1]
            if defect == "long_batch":
                return results + [deepcopy(results[0])]
            if defect == "missing_answer":
                results[0]["answers"].pop("authorized")
            elif defect == "nonfinite":
                results[0]["answers"]["authorized"]["noul"] = float("nan")
            return results

    report = diagnostic(Broken(), batch_sizes=(1, 2), repeats=1)
    assert report["passed"] is False
    assert report["cells"]["fixture"]["status"] == "failed"
    assert report["failures"]


@pytest.mark.parametrize("limits", [{"unknown": 1}, {"max_label_flip_rate": -1},
    {"max_probability_drift": float("nan")}, {"max_score_drift": float("inf")},
    {"max_p95_ms": -1}, {"max_p99_ms": float("inf")}, {"min_accuracy": 1.1},
    {"max_accuracy_drop": -1}, {"min_boundary_cases": -1}, {"boundary_margin": -0.1}])
def test_invalid_or_unknown_limits_cannot_disable_a_gate(limits):
    with pytest.raises(ValueError):
        diagnostic(limits=limits)


def test_missing_gold_labels_do_not_become_perfect_accuracy():
    document = corpus()
    for row in document["cases"]:
        row["expected"] = {}
    report = diagnostic(document=document, limits={"min_accuracy": 0.5}, repeats=1)
    assert report["passed"] is False
    assert report["failures"]


def test_boundary_coverage_counts_distinct_cases_not_questions():
    report = diagnostic(limits={"min_boundary_cases": 5}, repeats=1)
    assert len(corpus()["cases"]) == 4
    assert report["coverage"]["boundary_answers"] == 8
    assert report["passed"] is False
    assert any("boundary" in failure for failure in report["failures"])


def test_versioned_release_fixture_contains_all_primitives_and_family_splits():
    path = Path(__file__).resolve().parents[1] / "research/evals/qualification-v1.json"
    document = validate_corpus(json.loads(path.read_text(encoding="utf-8")))
    assert {row["split"] for row in document["cases"]} == {"training", "calibration", "evaluation"}
    evaluation = [row for row in document["cases"] if row["split"] == "evaluation"]
    assert {q["type"] for row in evaluation for q in row["questions"].values()} == {"choice", "score", "noul"}
    assert {row["variant"] for row in evaluation} >= {
        "base", "paraphrase", "distractor", "evidence_removed", "counterfactual", "boundary_candidate"}
    assert any(not row["expected"] and "boundary-candidate" in row["tags"] for row in evaluation)


def matrix(*, optional=False):
    return {"schema": "veya-qualification-matrix/1", "reference": "reference",
            "cells": [{"name": "reference", "checkpoint": "/local/reference", "backend": "eager",
                       "device": "cpu", "precision": "fp32", "required": True},
                      {"name": "candidate", "checkpoint": "/local/candidate", "backend": "eager",
                       "device": "cpu", "precision": "fp32", "required": not optional}],
            "batch_sizes": [1, 2], "repeats": 1,
            "limits": {"min_accuracy": 1.0, "min_boundary_cases": 1}}


class LocalFixtureAgent(FixtureRunner):
    backend = "eager"
    device = "cpu"
    dtype = "float32"
    amp_enabled = False
    hooks = ()

    def __init__(self):
        self.closed = False

    def predict(self, state, questions, model=None):
        assert not self.closed, "a closed qualification cell was used again"
        return super().predict(state, questions, model=model)

    def close(self):
        self.closed = True


def local_runner(agent=None):
    identity = {"verification": "local artifacts and runtime", "backend": "eager",
                "device": "cpu", "precision": "float32", "artifacts": {"fixture": "unit-test"}}
    return qualification._LocalRunner(agent or LocalFixtureAgent(), identity)


@pytest.fixture
def fixed_local_artifacts(monkeypatch):
    # These matrix tests use dummy agents. Keep their declared artifacts stable
    # and isolate source identity from other agents editing this shared workspace.
    monkeypatch.setattr(qualification, "_artifact_identity", lambda *args: {"fixture": "unit-test"})
    monkeypatch.setattr(qualification, "_source_identity", lambda: {"qualification.py": "fixture-source"})


def test_matrix_validation_returns_an_independent_document():
    original = matrix()
    validated = validate_matrix(original)
    assert validated == original
    validated["cells"][0]["checkpoint"] = "changed"
    assert original["cells"][0]["checkpoint"] == "/local/reference"


@pytest.mark.parametrize("field,value", [("backend", "unknown"), ("device", "auto"),
    ("device", "cuda:bad"), ("precision", "auto"), ("precision", "fp16"),
    ("required", "false"), ("checkpoint", ""), ("extra", "unreviewed")])
def test_malformed_matrix_cells_are_refused_before_loading(field, value):
    document = matrix()
    document["cells"][1][field] = value
    with pytest.raises(ValueError):
        validate_matrix(document)


@pytest.mark.parametrize("change", ["duplicate_name", "unknown_reference", "optional_reference",
                                   "empty_cells", "bad_schema", "bad_size", "unknown_threshold"])
def test_malformed_matrix_contract_cannot_produce_a_release_pass(change):
    document = matrix()
    if change == "duplicate_name":
        document["cells"][1]["name"] = "reference"
    elif change == "unknown_reference":
        document["reference"] = "missing"
    elif change == "optional_reference":
        document["cells"][0]["required"] = False
    elif change == "empty_cells":
        document["cells"] = []
    elif change == "bad_schema":
        document["schema"] = "veya-qualification-matrix/999"
    elif change == "bad_size":
        document["batch_sizes"] = [0, 2]
    elif change == "unknown_threshold":
        document["limits"] = {"misspelled_threshold": 0}
    with pytest.raises(ValueError):
        validate_matrix(document)


def test_cells_load_one_at_a_time_and_saved_reference_is_not_rerun(monkeypatch, fixed_local_artifacts):
    agents, runners, order = [], [], []

    def build(spec):
        assert all(agent.closed for agent in agents), "matrix retained a previous model"
        agent = LocalFixtureAgent()
        agents.append(agent)
        order.append(spec["name"])
        runner = local_runner(agent)
        runners.append(runner)
        return runner

    monkeypatch.setattr(qualification, "_build_cell", build)
    document = matrix()
    document["cells"].reverse()      # Reference order is a contract, not input-list order.
    report = run_matrix(corpus(), document)
    assert report["passed"] is True
    assert report["production_qualified"] is True
    assert order == ["reference", "candidate"]
    assert all(agent.closed for agent in agents)
    assert all(runner.agent is None for runner in runners)
    assert set(report["coverage"]["completed_cells"]) == {"reference", "candidate"}
    assert report["coverage"]["skipped_cells"] == []


@pytest.mark.parametrize("required", [False, True])
def test_unavailable_cells_have_explicit_coverage_and_required_failures_fail(monkeypatch, fixed_local_artifacts, required):
    reference = local_runner()
    reference_agent = reference.agent

    def build(spec):
        if spec["name"] == "candidate":
            raise qualification._Unavailable("fixture accelerator is unavailable")
        return reference

    monkeypatch.setattr(qualification, "_build_cell", build)
    document = matrix(optional=not required)
    report = run_matrix(corpus(), document)
    assert report["passed"] is (not required)
    assert report["production_qualified"] is (not required)
    assert report["cells"]["candidate"]["status"] == ("failed" if required else "skipped")
    assert "fixture accelerator is unavailable" in report["cells"]["candidate"]["error"]
    assert report["coverage"]["completed_cells"] == ["reference"]
    assert report["coverage"]["skipped_cells"] == ([] if required else ["candidate"])
    assert reference_agent.closed
    assert reference.agent is None


def test_unavailable_reference_never_becomes_a_pass_or_runs_candidates(monkeypatch, fixed_local_artifacts):
    calls = []

    def build(spec):
        calls.append(spec["name"])
        raise qualification._Unavailable("no reference hardware")

    monkeypatch.setattr(qualification, "_build_cell", build)
    report = run_matrix(corpus(), matrix())
    assert report["passed"] is False
    assert report["production_qualified"] is False
    assert report["cells"]["reference"]["status"] == "failed"
    assert report["coverage"]["completed_cells"] == []
    assert report["cells"]["candidate"]["status"] == "not_run"
    assert calls == ["reference"]


def test_arbitrary_loader_errors_fail_even_for_optional_cells(monkeypatch, fixed_local_artifacts):
    def build(spec):
        if spec["name"] == "candidate":
            raise ValueError("checkpoint artifact failed validation")
        return local_runner()

    monkeypatch.setattr(qualification, "_build_cell", build)
    report = run_matrix(corpus(), matrix(optional=True))
    assert report["passed"] is False
    assert report["production_qualified"] is False
    assert report["cells"]["candidate"]["status"] == "failed"
    assert report["coverage"]["skipped_cells"] == []


def test_agreement_without_explicit_quality_and_boundary_gates_is_not_production_qualification(monkeypatch, fixed_local_artifacts):
    monkeypatch.setattr(qualification, "_build_cell", lambda spec: local_runner())
    document = matrix()
    document.pop("limits")
    report = run_matrix(corpus(), document)
    assert report["passed"] is True
    assert report["production_qualified"] is False


@pytest.mark.parametrize("change", ["backend", "device", "precision", "hooks"])
def test_local_runner_refuses_runtime_identity_changes(change):
    runner = local_runner()
    runner._check()
    if change == "backend":
        runner.agent.backend = "compile"
    elif change == "device":
        runner.agent.device = "cuda:0"
    elif change == "precision":
        runner.agent.amp_enabled, runner.agent.dtype = True, "float16"
    else:
        runner.agent.hooks = [object()]
    with pytest.raises(ValueError):
        runner.predict(corpus()["cases"][0]["state"], QUESTIONS)


def test_runtime_fallback_during_a_call_is_detected_after_prediction():
    class Fallback(LocalFixtureAgent):
        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            self.backend = "compile"
            return result

    with pytest.raises(ValueError, match="identity"):
        local_runner(Fallback()).predict(corpus()["cases"][0]["state"], QUESTIONS)


def test_onnx_provider_change_is_detected_during_prediction():
    class Providers:
        def __init__(self):
            self.providers = ["CPUExecutionProvider"]

        def get_providers(self):
            return self.providers

    class ChangedProvider(LocalFixtureAgent):
        session = Providers()

        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            self.session.providers = ["UnexpectedExecutionProvider"]
            return result

    runner = qualification._LocalRunner(ChangedProvider(), {
        "backend": "onnx", "providers": ["CPUExecutionProvider"]})
    with pytest.raises(ValueError, match="providers"):
        runner.predict(corpus()["cases"][0]["state"], QUESTIONS)


@pytest.mark.parametrize("location", ["score_gold", "probability_limit", "evaluation_fraction", "calibration_fraction"])
def test_wide_integers_are_refused_as_numeric_inputs(location):
    wide = 10 ** 400
    document = corpus()
    with pytest.raises(ValueError):
        if location == "score_gold":
            document["cases"][0]["expected"]["priority"] = wide
            validate_corpus(document)
        elif location == "probability_limit":
            diagnostic(limits={"max_probability_drift": wide})
        elif location == "evaluation_fraction":
            split_families(document, evaluation_fraction=wide)
        else:
            split_families(document, calibration_fraction=wide)


@pytest.mark.parametrize("field", ["noul", "probability", "score", "answer_confidence"])
def test_wide_or_nonfinite_runtime_fields_fail_the_report(field):
    class InvalidNumbers(FixtureRunner):
        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            if field == "noul":
                result["answers"]["authorized"]["noul"] = 10 ** 400
            elif field == "probability":
                result["answers"]["route"]["probabilities"]["allow"] = 10 ** 400
            elif field == "score":
                result["answers"]["priority"]["score"] = 10 ** 400
            else:
                result["answers"]["route"]["answer_confidence"] = float("nan")
            return result

    report = diagnostic(InvalidNumbers(), repeats=1)
    assert report["passed"] is False
    assert report["cells"]["fixture"]["status"] == "failed"
    assert report["failures"]


@pytest.mark.parametrize("values", [None, [], [0], [True], [65]])
def test_invalid_programmatic_concurrency_profiles_are_refused(values):
    with pytest.raises(ValueError):
        diagnostic(concurrencies=values)


@pytest.mark.parametrize("values", [None, [], [0], [True], [65]])
def test_invalid_matrix_concurrency_profiles_are_refused(values):
    document = matrix()
    document["concurrencies"] = values
    with pytest.raises(ValueError):
        validate_matrix(document)


def test_concurrency_profiles_are_recorded_and_preserve_decisions():
    report = diagnostic(batch_sizes=(1, 2), repeats=1, concurrencies=(1, 2))
    assert report["passed"] is True
    assert report["concurrencies"] == [1, 2]
    runs = report["cells"]["fixture"]["runs"]
    assert {run["concurrency"] for run in runs} == {1, 2}
    assert all(run["comparison"]["label_flip_rate"] == 0 for run in runs)


def test_local_concurrency_serializes_inference_and_measures_gate_wait(monkeypatch):
    entered, queued = threading.Event(), threading.Event()
    mutex = threading.Lock()
    clock = {"now": 0.0}
    calls = {"started": 0, "active": 0, "maximum": 0, "seen": []}

    class Waiting(LocalFixtureAgent):
        def predict(self, state, questions, model=None):
            with mutex:
                calls["active"] += 1
                calls["maximum"] = max(calls["maximum"], calls["active"])
                calls["seen"].append(state["case"])
                first = len(calls["seen"]) == 1
                clock["now"] += 5.0 if first else 3.0
            if first:
                entered.set()
                assert queued.wait(2), "the second request never entered the caller queue"
                with mutex:
                    clock["now"] += 7.0
            result = super().predict(state, questions, model=model)
            with mutex:
                calls["active"] -= 1
            return result

    class ObservedGate:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            with mutex:
                calls["started"] += 1
                number = calls["started"]
            if number == 2:
                assert entered.wait(2)
                queued.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    runner = local_runner(Waiting())
    runner._lock = ObservedGate()
    monkeypatch.setattr(qualification.time, "perf_counter", lambda: clock["now"])
    cases = corpus()["cases"][:2]
    observation = qualification._observe(runner, cases, 1, "ordered", concurrency=2)
    assert calls["maximum"] == 1
    assert calls["started"] == 2
    second = next(i for i, case in enumerate(cases) if case["state"]["case"] == calls["seen"][1])
    # A deterministic virtual clock gives this request 3 seconds of service and at
    # least 7 seconds waiting for the single inference gate. No wall-clock sleep.
    assert observation["latencies_ms"][second] >= 10_000.0


def build_probe_cell(monkeypatch, *, precision, actual_autocast=None, keyword=False):
    torch = pytest.importorskip("torch")
    import laya.agent as agent_module
    import laya.consistency as consistency

    class Probe(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids

    class ProbeAgent(LocalFixtureAgent):
        def __init__(self, checkpoint, **kwargs):
            super().__init__()
            self.device = torch.device("cpu")
            self.amp_enabled = precision == "bf16"
            self.dtype = torch.bfloat16 if self.amp_enabled else torch.float32
            self.model = Probe()

        def predict(self, state, questions, model=None):
            tokens = torch.zeros((1, 3), dtype=torch.int64)
            context = (torch.autocast("cpu", dtype=torch.bfloat16)
                       if actual_autocast == "bf16" else nullcontext())
            with context:
                if keyword:
                    self.model(input_ids=tokens)
                else:
                    self.model(tokens)
            return super().predict(state, questions, model=model)

    monkeypatch.setattr(agent_module, "Agent", ProbeAgent)
    monkeypatch.setattr(qualification, "_artifact_identity", lambda *args: {"fixture": "stable"})
    monkeypatch.setattr(consistency, "_fingerprint", lambda agent: ["probe-model-v1"])
    return qualification._build_cell({"checkpoint": "/local/probe", "backend": "eager",
                                      "device": "cpu", "precision": precision})


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_factory_witness_records_positional_and_keyword_forward_inputs(monkeypatch, precision, keyword):
    runner = build_probe_cell(monkeypatch, precision=precision,
                              actual_autocast="bf16" if precision == "bf16" else None, keyword=keyword)
    agent = runner.agent
    model = agent.model
    try:
        result = runner.predict(corpus()["cases"][0]["state"], QUESTIONS)
        assert result["answers"]["route"]["choice"] == "allow"
        assert runner._local.forwards == [{"device": "cpu", "question_rows": 1,
                                          "precision": "bfloat16" if precision == "bf16" else "float32"}]
        assert runner.agent.model._forward_pre_hooks
    finally:
        runner.close()
    assert runner.agent is None
    assert agent.closed
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("requested,actual", [("fp32", "bf16"), ("bf16", None)])
def test_factory_witness_rejects_actual_precision_fallback_despite_unchanged_flags(monkeypatch, requested, actual):
    runner = build_probe_cell(monkeypatch, precision=requested, actual_autocast=actual)
    try:
        with pytest.raises(ValueError, match="unexpected.*precision"):
            runner.predict(corpus()["cases"][0]["state"], QUESTIONS)
    finally:
        runner.close()


@pytest.mark.parametrize("target,optional", [("reference", False), ("candidate", False), ("candidate", True)])
def test_changed_checkpoint_artifacts_fail_even_optional_cells(monkeypatch, fixed_local_artifacts, target, optional):
    runners = []

    def build(spec):
        runner = local_runner()
        runners.append((runner, runner.agent))
        return runner

    monkeypatch.setattr(qualification, "_build_cell", build)
    monkeypatch.setattr(qualification, "_artifact_identity", lambda path, *args:
                        {"fixture": "changed" if str(path).endswith(target) else "unit-test"})
    report = run_matrix(corpus(), matrix(optional=optional))
    assert report["passed"] is False
    assert report["production_qualified"] is False
    assert report["cells"][target]["status"] == "failed"
    assert "artifacts changed" in report["cells"][target]["error"]
    assert report["coverage"]["skipped_cells"] == []
    if target == "reference":
        assert report["cells"]["candidate"]["status"] == "not_run"
    assert all(agent.closed and runner.agent is None for runner, agent in runners)


def test_changed_runtime_source_refuses_a_release_pass(monkeypatch, fixed_local_artifacts):
    identities = iter([{"qualification.py": "before"}, {"qualification.py": "after"}])
    monkeypatch.setattr(qualification, "_source_identity", lambda: next(identities))
    monkeypatch.setattr(qualification, "_build_cell", lambda spec: local_runner())
    report = run_matrix(corpus(), matrix())
    assert report["passed"] is False
    assert report["production_qualified"] is False
    assert report["source_changed_during_run"] is True
    assert any("source files changed" in failure for failure in report["failures"])


def test_close_releases_real_agent_backend_cycles_even_with_automatic_gc_disabled():
    torch = pytest.importorskip("torch")
    from laya.agent import Agent
    from laya.backends.base import Backend

    agent = object.__new__(Agent)
    agent.model = torch.nn.Identity()
    agent._backend, agent._fast = Backend(agent), None
    agent._backend.install()
    model = weakref.ref(agent.model)
    backend = weakref.ref(agent._backend)
    runner = qualification._LocalRunner(agent, {"backend": "eager"})
    automatic_gc = gc.isenabled()
    gc.disable()
    try:
        runner.close()
        assert runner.agent is None
        assert agent.model is None
        assert agent._backend is None
        assert model() is None
        assert backend() is None
        runner.close()             # Repeated closure must not recreate or retain resources.
    finally:
        if automatic_gc:
            gc.enable()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_partial_backend_setup_is_disposed_without_masking_the_original_failure(monkeypatch, cleanup_failure):
    pytest.importorskip("torch")
    import laya.agent as agent_module

    original = RuntimeError("original backend failure")
    agents = []

    class Partial(LocalFixtureAgent):
        def __init__(self, checkpoint, **kwargs):
            super().__init__()
            self.model = object()
            self.deaccelerated = False
            agents.append(self)

        def set_backend(self, name, strict=False):
            assert name == "compile" and strict is True
            raise original

        def deaccelerate(self):
            self.deaccelerated = True

        def close(self):
            self.model = None
            super().close()
            if cleanup_failure:
                raise ValueError("injected cleanup failure")

    monkeypatch.setattr(agent_module, "Agent", Partial)
    monkeypatch.setattr(qualification, "_artifact_identity", lambda *args: {"fixture": "stable"})
    names = ("LAYA_CPU_AMP", "LAYA_CUDA_AMP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    before = {name: os.environ.get(name) for name in names}
    with pytest.raises(RuntimeError) as failure:
        qualification._build_cell({"checkpoint": "/local/probe", "backend": "compile",
                                   "device": "cpu", "precision": "fp32"})
    assert failure.value is original
    assert len(agents) == 1
    assert agents[0].closed and agents[0].deaccelerated and agents[0].model is None
    assert {name: os.environ.get(name) for name in names} == before
    if cleanup_failure:
        assert isinstance(failure.value.__cause__, ValueError)
        assert "cleanup failure" in str(failure.value.__cause__)


def test_matrix_cleanup_failure_stops_before_loading_the_next_model(monkeypatch, fixed_local_artifacts):
    class FailedCleanup(LocalFixtureAgent):
        def close(self):
            super().close()
            raise RuntimeError("injected cleanup failure")

    agent = FailedCleanup()
    runner = local_runner(agent)
    loaded = []

    def build(spec):
        loaded.append(spec["name"])
        return runner

    monkeypatch.setattr(qualification, "_build_cell", build)
    report = run_matrix(corpus(), matrix())
    assert report["passed"] is False
    assert report["production_qualified"] is False
    assert report["cells"]["reference"]["status"] == "failed"
    assert report["cells"]["candidate"]["status"] == "not_run"
    assert loaded == ["reference"]
    assert any("cleanup failed" in failure for failure in report["failures"])
    assert agent.closed and runner.agent is None


def test_other_cleanup_steps_run_even_when_deacceleration_fails():
    class FailedDeacceleration(LocalFixtureAgent):
        def deaccelerate(self):
            raise RuntimeError("injected deacceleration failure")

    agent = FailedDeacceleration()
    runner = local_runner(agent)
    with pytest.raises(RuntimeError, match="deacceleration failure"):
        runner.close()
    assert agent.closed and runner.agent is None


def test_onnx_session_is_released_when_agent_has_no_context_manager():
    class SessionOnly:
        def __init__(self):
            self.session = object()

    agent = SessionOnly()
    runner = qualification._LocalRunner(agent, {"backend": "onnx"})
    runner.close()
    assert agent.session is None
    assert runner.agent is None


def test_explicit_backend_unavailability_is_classified_after_partial_agent_cleanup(monkeypatch):
    pytest.importorskip("torch")
    import laya.agent as agent_module
    from laya.backends.base import BackendUnavailable

    original = BackendUnavailable("injected compiler feature unavailable")
    agents = []

    class Unavailable(LocalFixtureAgent):
        def __init__(self, checkpoint, **kwargs):
            super().__init__()
            self.deaccelerated = False
            agents.append(self)

        def set_backend(self, name, strict=False):
            raise original

        def deaccelerate(self):
            self.deaccelerated = True

    monkeypatch.setattr(agent_module, "Agent", Unavailable)
    monkeypatch.setattr(qualification, "_artifact_identity", lambda *args: {"fixture": "stable"})
    names = ("LAYA_CPU_AMP", "LAYA_CUDA_AMP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    before = {name: os.environ.get(name) for name in names}
    with pytest.raises(qualification._Unavailable, match="compiler feature unavailable") as failure:
        qualification._build_cell({"checkpoint": "/local/probe", "backend": "compile",
                                   "device": "cpu", "precision": "fp32"})
    assert failure.value.__cause__ is original
    assert len(agents) == 1 and agents[0].closed and agents[0].deaccelerated
    assert {name: os.environ.get(name) for name in names} == before


def test_missing_onnx_runtime_is_explicit_unavailability(monkeypatch):
    original_import = builtins.__import__

    def without_onnx_runtime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("injected missing ONNX dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(qualification, "_artifact_identity", lambda *args: {"fixture": "stable"})
    monkeypatch.setattr(builtins, "__import__", without_onnx_runtime)
    with pytest.raises(qualification._Unavailable, match="ONNX runtime dependencies") as failure:
        qualification._build_cell({"checkpoint": "/local/probe", "backend": "onnx", "device": "cpu",
                                   "precision": "graph", "onnx_path": "/local/probe.onnx"})
    assert isinstance(failure.value.__cause__, ImportError)


class OperationalRunner(FixtureRunner):
    def predict(self, state, questions, model=None):
        result = super().predict(state, questions, model=model)
        for answer in result["answers"].values():
            answer.update(action={"act_probability": 0.51}, low_confidence=False, abstention="accepted")
        return result


def set_operational_field(answer, field, value):
    if field == "act_probability":
        answer["action"] = {"act_probability": value}
    else:
        answer[field] = value


@pytest.mark.parametrize("field", ["confidence", "answer_confidence", "act_probability"])
def test_batch_operational_drift_fails_even_when_primary_probabilities_are_stable(field):
    class Changed(OperationalRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            set_operational_field(results[0]["answers"]["route"], field, 0.99)
            return results

    report = diagnostic(Changed(), repeats=1)
    assert report["passed"] is False
    runs = report["cells"]["fixture"]["runs"]
    assert all(run["comparison"]["max_probability_drift"] == 0.0 for run in runs)
    metric = "max_action_drift" if field == "act_probability" else "max_confidence_drift"
    assert any(metric in failure for failure in report["failures"])


@pytest.mark.parametrize("field", ["confidence", "answer_confidence", "act_probability"])
@pytest.mark.parametrize("change", ["missing", "new"])
def test_operational_field_presence_is_part_of_the_execution_contract(field, change):
    key = "action" if field == "act_probability" else field

    class Changed(OperationalRunner):
        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            if change == "new":
                result["answers"]["route"].pop(key)
            return result

        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            if change == "missing":
                results[0]["answers"]["route"].pop(key)
            else:
                set_operational_field(results[0]["answers"]["route"], field, 0.51)
            return results

    report = diagnostic(Changed(), repeats=1)
    assert report["passed"] is False
    assert any("field presence" in failure for failure in report["failures"])


@pytest.mark.parametrize("field,value", [("low_confidence", True), ("abstention", "abstained")])
def test_changed_review_or_abstention_flags_fail_the_matrix(field, value):
    class Changed(OperationalRunner):
        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            results[0]["answers"]["route"][field] = value
            return results

    report = diagnostic(Changed(), repeats=1)
    assert report["passed"] is False
    assert any("operational_flag_flip_rate" in failure for failure in report["failures"])


@pytest.mark.parametrize("field", ["confidence", "answer_confidence", "act_probability"])
@pytest.mark.parametrize("value", [-0.01, 1.01, True, float("nan")])
def test_invalid_operational_probabilities_fail_qualification(field, value):
    class Invalid(OperationalRunner):
        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            set_operational_field(result["answers"]["route"], field, value)
            return result

    report = diagnostic(Invalid(), repeats=1)
    assert report["passed"] is False
    assert report["cells"]["fixture"]["status"] == "failed"


@pytest.mark.parametrize("field,value", [("low_confidence", 0), ("low_confidence", "false"),
                                         ("abstention", None), ("abstention", 0)])
def test_operational_flags_have_strict_types(field, value):
    class Invalid(OperationalRunner):
        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            result["answers"]["route"][field] = value
            return result

    assert diagnostic(Invalid(), repeats=1)["passed"] is False


@pytest.mark.parametrize("field,value", [("low_confidence", False), ("abstention", "accepted")])
@pytest.mark.parametrize("change", ["missing", "new"])
def test_review_flag_field_presence_is_part_of_the_execution_contract(field, value, change):
    class Changed(OperationalRunner):
        def predict(self, state, questions, model=None):
            result = super().predict(state, questions, model=model)
            if change == "new":
                result["answers"]["route"].pop(field)
            return result

        def predict_batch(self, states, questions, model=None, batch_size=None):
            results = super().predict_batch(states, questions, model=model, batch_size=batch_size)
            if change == "missing":
                results[0]["answers"]["route"].pop(field)
            else:
                results[0]["answers"]["route"][field] = value
            return results

    report = diagnostic(Changed(), repeats=1)
    assert report["passed"] is False
    assert any("field presence" in failure for failure in report["failures"])


def test_factory_records_the_actual_torch_numerical_execution_policy(monkeypatch):
    runner = build_probe_cell(monkeypatch, precision="fp32")
    try:
        settings = runner.identity["math_settings"]
        assert settings == qualification._torch_math_settings("cpu")
        assert set(settings) >= {"float32_matmul_precision", "deterministic_algorithms", "mkldnn_enabled"}
    finally:
        runner.close()


def test_local_runner_rejects_numerical_policy_changes(monkeypatch):
    runner = local_runner()
    runner.identity["math_settings"] = {"execution": "fixed"}
    monkeypatch.setattr(qualification, "_torch_math_settings", lambda device: {"execution": "fixed"})
    runner._check()
    monkeypatch.setattr(qualification, "_torch_math_settings", lambda device: {"execution": "changed"})
    try:
        with pytest.raises(ValueError, match="numerical execution policy"):
            runner.predict(corpus()["cases"][0]["state"], QUESTIONS)
    finally:
        runner.close()


def test_modern_mkldnn_precision_is_recorded_without_legacy_api_calls(monkeypatch):
    torch = pytest.importorskip("torch")
    matmul = getattr(torch.backends.mkldnn, "matmul", None)
    if getattr(torch.backends, "fp32_precision", None) is None or getattr(matmul, "fp32_precision", None) is None:
        pytest.skip("this Torch release has no per-backend fp32 precision API")
    before = matmul.fp32_precision

    def forbidden():
        raise AssertionError("modern numerical policy inspection called a legacy API")

    monkeypatch.setattr(torch, "get_float32_matmul_precision", forbidden)
    try:
        matmul.fp32_precision = "bf16"
        settings = qualification._torch_math_settings("cpu")
        assert settings["float32_matmul_precision"] == "per-backend"
        assert settings["mkldnn_matmul_fp32_precision"] == "bf16"
    finally:
        matmul.fp32_precision = before


def test_legacy_torch_precision_inspection_remains_supported(monkeypatch):
    calls = []

    def legacy_precision():
        calls.append("legacy")
        return "highest"

    legacy = SimpleNamespace(
        backends=SimpleNamespace(mkldnn=SimpleNamespace(enabled=True)),
        get_float32_matmul_precision=legacy_precision,
        are_deterministic_algorithms_enabled=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", legacy)
    assert qualification._torch_math_settings("cpu") == {
        "float32_matmul_precision": "highest", "deterministic_algorithms": False, "mkldnn_enabled": True}
    assert calls == ["legacy"]
