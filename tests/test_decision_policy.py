"""Generic decision-policy mathematics and real Torch/ONNX call-path integration, without weights."""
import concurrent.futures
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.agent import Agent, load  # noqa: E402
from laya.decision_policy import (  # noqa: E402
    CONTENT_FREE_PROBES, MAX_BALANCED_OPTIONS, _FORWARD_TOKEN_BUDGET, resolve_decision_policy,
)
from laya.inference_common import QTYPES  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402


QUESTIONS = {
    "category": {"type": "choice", "instructions": "Choose the category.",
                 "criteria": {"alpha": "First category", "beta": "Second category", "gamma": "Third category"}},
    "level": {"type": "score", "instructions": "Choose the level.", "criteria": ["low", "medium", "high"]},
    "holds": {"type": "noul", "instructions": "Does this hold?", "labels": {"false": "No", "true": "Yes"}},
}


class Tokenizer:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [9] * len(text.split())
        return {"input_ids": ids[:max_length] if truncation else ids}

    def decode(self, ids):
        return " ".join("word" for _ in ids)


def _fixture(kind, policy="context-balanced-v1"):
    cls = Agent if kind == "torch" else ONNXAgent
    agent = object.__new__(cls)
    agent.cfg = {"max_len": 64, "head_max_len": 28}
    agent.model_id = "controlled-local-checkpoint"
    agent.tok = Tokenizer()
    agent.temperature = [1.3, 1.1, 1.7]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {"zz": {"temperature": [1.2, 1.2, 1.2], "temperature_by_options": {}}}
    agent.binning_map = None
    agent.decision_policy = policy
    agent.parallel_options = False
    agent.calls = []

    def encode(self, state, ids, internal, max_len=None, head_max_len=None):
        probe = isinstance(state, str) and state in CONTENT_FREE_PROBES
        text = state if isinstance(state, str) else json.dumps(state)
        count_state = len(text.split())
        items = []
        for qid in ids:
            q = internal[qid]
            count = 2 if q["t"] == "noul" else len(q["crit"])
            order = q.get("option_order", list(range(count)))
            items.append({"ids": [100 if probe else 101] + [20 + i for i in order] + [2],
                          "markers": list(range(1, count + 1)), "qtype": QTYPES[q["t"]],
                          "options": {"options": count, "options_distinct": count, "tokens_per_option": None},
                          "state_stats": {"state_tokens": count_state, "state_tokens_used": count_state,
                                          "state_tokens_dropped": 0, "truncated": False}})
        return items

    agent._encode_state = types.MethodType(encode, agent)

    def forward(inputs):
        def array(value):
            return value.numpy() if hasattr(value, "numpy") else value
        mask = array(inputs["marker_mask"])
        markers, ids = array(inputs["marker_pos"]), array(inputs["input_ids"])
        agent.calls.append((len(ids), len(ids[0])))
        labels = np.take_along_axis(ids, markers, axis=1) - 20
        logits = np.array([-0.8, 0.3, 0.7] + [0.0] * 13)[np.clip(labels, 0, 15)]
        logits += np.arange(mask.shape[1])[None, :] * 0.4
        logits += ((ids[:, :1] == 101) & (labels == 0)) * 2.0
        logits = np.where(mask, logits, -1e4)
        act = np.tile([0.75, 0.25], (len(ids), 1))
        return logits, act

    if kind == "torch":
        agent._forward = forward
    else:
        agent.session = types.SimpleNamespace(run=lambda names, inputs: [forward(inputs)[0],
            np.tile(np.log([0.75, 0.25]), (len(inputs["input_ids"]), 1))])
    return agent


@pytest.mark.parametrize("value", [None, "canonical", "context-balanced-v1"])
def test_policy_resolution(value):
    assert resolve_decision_policy(value) == (value or "canonical")
    assert resolve_decision_policy(None, {"decision_policy": "context-balanced-v1"}) == "context-balanced-v1"
    assert resolve_decision_policy("canonical", {"decision_policy": "bad"}) == "canonical"


@pytest.mark.parametrize("value", [False, 0, "", "balanced", {}, []])
def test_invalid_policy(value):
    with pytest.raises(ValueError, match="decision_policy"):
        resolve_decision_policy(value)


@pytest.mark.parametrize("factory", [Agent, ONNXAgent, load])
def test_invalid_constructor_policy_never_resolves_or_downloads_a_checkpoint(factory):
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("download reached")), \
            patch("laya.revisions.resolve_revision", side_effect=AssertionError("revision reached")):
        with pytest.raises(ValueError, match="decision_policy"):
            factory("review/policy-validation-probe", decision_policy="invalid-policy")


def test_policy_configuration_does_not_import_numerical_runtime():
    script = """
import sys
for name in ('numpy', 'torch', 'transformers', 'onnxruntime', 'tokenizers'):
    sys.modules[name] = None
from laya.decision_policy import resolve_decision_policy
assert resolve_decision_policy('context-balanced-v1') == 'context-balanced-v1'
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], check=True)


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_canonical_results_unchanged(kind):
    agent = _fixture(kind, "canonical")
    expected = agent._infer_batch(["actual"], QUESTIONS)
    actual = agent.predict_batch(["actual"], QUESTIONS)
    assert actual == expected
    assert "decision_policy" not in actual[0]["usage"]
    assert all("_decision_probabilities" not in value and "confidence_calibrated" not in value
               for value in actual[0]["answers"].values())


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_policy_matches_independent_probability_ratio_and_reports_all_work(kind):
    agent = _fixture(kind)
    original = copy.deepcopy(QUESTIONS)
    expected = {}
    for qid, question in QUESTIONS.items():
        count = 2 if question["type"] == "noul" else len(question["criteria"])
        rows = {r: dict(question, option_order=list(range(r, count)) + list(range(r))) for r in range(count)}
        probes = agent._infer_batch(list(CONTENT_FREE_PROBES), rows, full_precision=True)
        actual = agent._infer_batch(["actual"], rows, full_precision=True)[0]
        prior = np.mean([row["answers"][r]["_decision_probabilities"] for row in probes for r in rows], axis=0)
        values = np.mean([actual["answers"][r]["_decision_probabilities"] for r in rows], axis=0) / prior
        expected[qid] = values / sum(values)
    agent.calls.clear()
    result = agent.predict("actual", QUESTIONS)
    assert QUESTIONS == original
    assert result["answers"]["category"]["choice"] == "alpha"
    for qid, answer in result["answers"].items():
        actual = [1 - answer["noul"], answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())
        np.testing.assert_allclose(actual, expected[qid], atol=0.000051, rtol=0)
        assert answer["decision_policy"] == "context-balanced-v1"
        assert answer["confidence_calibrated"] is False
        assert answer["action"]["act_probability"] == 0.75
        assert answer["answer_confidence"] == round(float(max(expected[qid])), 4)
        assert "_decision_probabilities" not in answer
    usage = result["usage"]
    assert usage["policy_variant_rows"] == 8
    assert usage["policy_probe_rows"] == 24
    assert sum(rows for rows, _ in agent.calls) == 32
    assert usage["input_tokens"] == sum(rows * length for rows, length in agent.calls)
    assert usage["input_tokens"] == usage["policy_rotation_tokens"] + usage["policy_probe_tokens"]


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_batch_reuses_only_call_local_prior_and_preserves_inputs(kind):
    agent = _fixture(kind)
    states = ["actual", {"note": "actual"}, [{"text": "actual"}]]
    results = agent.predict_batch(states, QUESTIONS, batch_size=2, sort_by_length=True)
    for state, result in zip(states, results):
        assert result["answers"] == agent.predict(state, QUESTIONS)["answers"]
    assert results[0]["usage"]["policy_probe_tokens"] > 0
    assert all(row["usage"]["policy_probe_tokens"] == 0 for row in results[1:])
    assert all(row["usage"]["policy_variant_rows"] == 8 for row in results)


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_hooks_run_once_and_gate_corrected_confidence(kind):
    agent = _fixture(kind)
    events = []
    def start(ctx):
        events.append("start")
        ctx.states = ["actual"]
    def end(ctx):
        events.append("end")
        assert ctx.results[0]["answers"]["category"]["decision_policy"] == "context-balanced-v1"
    result = agent.predict("original", QUESTIONS, on_predict_start=start, on_predict_end=end, min_confidence=1.0)
    assert events == ["start", "end"]
    assert all(answer["low_confidence"] is True for answer in result["answers"].values())


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_start_hook_skip_does_not_run_policy(kind):
    agent = _fixture(kind)
    answer = {"model": "hook", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
    assert agent.predict("actual", QUESTIONS, on_predict_start=lambda ctx: ctx.skip([answer])) == answer
    assert agent.calls == []


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_option_limit_validates_all_questions_before_any_forward(kind):
    agent = _fixture(kind)
    questions = {**QUESTIONS, "too_many": {"type": "choice", "instructions": "Choose",
        "criteria": {str(i): str(i) for i in range(MAX_BALANCED_OPTIONS + 1)}}}
    with pytest.raises(ValueError, match="at most 16"):
        agent.predict("actual", questions)
    assert agent.calls == []


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_variant_rows_and_tokenized_lookahead_are_bounded(kind):
    agent = _fixture(kind)
    question = {"many": {"type": "choice", "instructions": "Choose",
                        "criteria": {str(i): str(i) for i in range(16)}}}
    results = agent.predict_batch(["actual"] * 11, question, batch_size=100, sort_by_length=True)
    assert len(results) == 11
    assert max(rows for rows, _ in agent.calls) <= 64
    assert sum(rows for rows, _ in agent.calls) == 16 * (11 + 3)


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_length_sort_reduces_padding_without_reordering_results(kind):
    agent = _fixture(kind)
    original = agent._encode_state
    def varied(state, *args, **kwargs):
        rows = original(state, *args, **kwargs)
        if state.startswith("word"):
            for row in rows:
                row["ids"].extend([9] * len(state.split()))
        return rows
    agent._encode_state = varied
    states = ["word " * width for width in (1, 10, 2, 12, 3, 14)]
    question = {"q": QUESTIONS["category"]}
    ordinary = agent.predict_batch(states, question, batch_size=2)
    padded = sum(rows * length for rows, length in agent.calls)
    agent.calls.clear()
    sorted_rows = agent.predict_batch(states, question, batch_size=2, sort_by_length=True)
    assert sorted_rows == ordinary
    assert sum(rows * length for rows, length in agent.calls) < padded
    assert max(rows for rows, _ in agent.calls) <= 6


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_forward_token_budget_splits_rotations_and_probes_without_changing_results(kind):
    agent = _fixture(kind)
    expected = agent.predict_batch(["actual", "other actual"], QUESTIONS, max_len=48)
    agent.calls.clear()
    token = _FORWARD_TOKEN_BUDGET.set(48)
    try:
        actual = agent.predict_batch(["actual", "other actual"], QUESTIONS, max_len=48)
        assert actual == expected
        assert len(agent.calls) == 40
        assert max(rows for rows, _ in agent.calls) == 1
        agent.calls.clear()
        with pytest.raises(ValueError, match="exceeds the forward token budget"):
            agent.predict("actual", QUESTIONS)
        assert agent.calls == []
    finally:
        _FORWARD_TOKEN_BUDGET.reset(token)
    assert agent.predict("actual", QUESTIONS)["answers"] == expected[0]["answers"]


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_language_override_and_explicit_option_order(kind):
    agent = _fixture(kind)
    question = {"q": dict(QUESTIONS["category"], option_order=[2, 0, 1])}
    ordinary = agent.predict("actual", question)
    language = agent.predict("actual", question, lang="zz-ZZ")
    assert set(language["answers"]["q"]["probabilities"]) == {"alpha", "beta", "gamma"}
    assert language["answers"] != ordinary["answers"]
    assert question["q"]["option_order"] == [2, 0, 1]


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_old_binning_map_is_not_applied_to_new_policy(kind):
    agent = _fixture(kind)
    agent.binning_map = {"invalid-for-new-policy": True}
    module = "laya.agent" if kind == "torch" else "laya.calibration"
    with patch(module + ".apply_binning_map", side_effect=AssertionError("old map applied")):
        assert agent.predict("actual", QUESTIONS)["answers"]["category"]["confidence_calibrated"] is False


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_empty_and_missing_inputs(kind):
    agent = _fixture(kind)
    assert agent.predict_batch([], QUESTIONS) == []
    assert agent.predict("actual", {})["answers"] == {}
    assert agent.calls == []
    with pytest.raises(TypeError, match="must not be None"):
        agent.predict(None, QUESTIONS)
    with pytest.raises(TypeError, match="expects a list"):
        agent.predict_batch("actual", QUESTIONS)


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_concurrent_calls_and_windowed_long_honor_policy(kind):
    agent = _fixture(kind)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda _: agent.predict("actual", QUESTIONS), range(6)))
    assert all(row["answers"] == rows[0]["answers"] for row in rows)
    long = agent.predict_long(" ".join(["word"] * 70), QUESTIONS, window=12, stride=6, batch_size=2)
    assert long["usage"]["windows"] > 1
    assert all(answer["decision_policy"] == "context-balanced-v1" for answer in long["answers"].values())
    assert all(answer["confidence_calibrated"] is False for answer in long["answers"].values())


@pytest.mark.parametrize("kind", ["torch", "onnx"])
def test_nonfinite_model_results_fail_without_canonical_fallback(kind):
    agent = _fixture(kind)
    original = agent._infer_batch
    def invalid(*args, **kwargs):
        results = original(*args, **kwargs)
        results[0]["answers"][0]["_decision_probabilities"][0] = float("nan")
        return results
    agent._infer_batch = invalid
    with pytest.raises(ValueError, match="invalid full-precision"):
        agent.predict("actual", QUESTIONS)


def test_ratio_machine_floor_is_finite_without_tuned_smoothing():
    from laya.decision_policy import _correct
    values = _correct(np.array([0., 1.]), np.array([0., 1.]))
    np.testing.assert_array_equal(values, [.5, .5])
    assert np.isfinite(_correct(np.array([1., 0.]), np.array([0., 1.]))).all()


def test_load_forwards_policy_to_both_backends():
    with patch("laya.agent.Agent") as cls:
        load("unused/local-model", decision_policy="context-balanced-v1")
        assert cls.call_args.kwargs["decision_policy"] == "context-balanced-v1"
    with patch("laya.onnx_agent.ONNXAgent") as cls:
        load("unused/local-model", backend="onnx", decision_policy="context-balanced-v1")
        assert cls.call_args.kwargs["decision_policy"] == "context-balanced-v1"
