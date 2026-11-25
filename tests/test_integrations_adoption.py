"""Regression checks for optional adapters and artifact validation; no external services."""
import io
import json
from types import SimpleNamespace

import pytest

from veya._eval_policy import load_policy
from veya.confidence import apply_confidence_gate
from veya.evals import EvalError
from veya.integrations import VeyaLowConfidenceError, agno, crewai, llamaindex
from veya.integrations._controls import require_router_controls


@pytest.mark.parametrize("controls", [{"task": "typed-decisions"}, {"lang_guess": "fr"},
                                     {"task": "english", "lang_guess": ""}])
def test_agno_remote_routing_controls_reach_the_wire(monkeypatch, controls):
    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(b'{"answers":{}}')

    monkeypatch.setattr(agno.urllib.request, "build_opener",
                        lambda *args: SimpleNamespace(open=open_request))
    assert agno._execute_decision("hello", {}, base_url="http://127.0.0.1:8000", **controls) == {"answers": {}}
    payload = json.loads(requests[0][0].data)
    assert {key: payload[key] for key in controls} == controls


@pytest.mark.parametrize("choice", [None, "agent_00", "agent_bad", "agent_-1", "agent_2", "unknown"])
def test_agno_rejects_unknown_choices_before_fallback(choice):
    runner = SimpleNamespace(predict=lambda *args, **kwargs: {
        "answers": {"delegation": {"type": "choice", "choice": choice, "confidence": 0.1}}})
    adapter = agno.VeyaAgnoRouter(agent=runner, confidence_threshold=0.9, fallback_agent_index=0)
    with pytest.raises(RuntimeError, match="unknown agent choice"):
        adapter.route("hello", ["first", "second"])


def _core_gated_runner(confidence):
    def predict(state, questions, *, min_confidence):
        qid, question = next(iter(questions.items()))
        keys = list(question["criteria"])
        answer = {"type": "choice", "choice": keys[0], "probabilities": {keys[0]: 0.6, keys[1]: 0.4}}
        if confidence is not None:
            answer["answer_confidence"] = confidence
        result = {"answers": {qid: answer}}
        apply_confidence_gate([result], min_confidence)
        return result

    return SimpleNamespace(predict=predict)


@pytest.mark.parametrize("router", [agno.VeyaAgnoRouter, crewai.VeyaCrewRouter])
@pytest.mark.parametrize("confidence, status", [(0.6, "abstained"), (None, "unevaluated"), (0.9, "passed")])
def test_framework_routers_honor_core_abstention(router, confidence, status):
    runner = _core_gated_runner(confidence)
    candidates = ["first", "backup"]
    route = router(agent=runner, min_confidence={"choice:2": 0.8}, fallback_agent_index=1)
    decision = route.route("hello", candidates)
    assert decision.agent == ("first" if status == "passed" else "backup")
    assert decision.raw_decision["answers"]["delegation"]["abstention"] == status
    assert decision.confidence == confidence
    if status != "passed":
        route_without_fallback = router(agent=runner, min_confidence=0.8)
        with pytest.raises(VeyaLowConfidenceError) as exc:
            route_without_fallback.route("hello", candidates)
        assert exc.value.confidence == confidence
        assert exc.value.threshold == 0.8


@pytest.mark.parametrize("confidence", [0.6, None])
def test_llamaindex_selectors_and_query_router_honor_core_abstention(confidence):
    runner = _core_gated_runner(confidence)
    choices = [llamaindex.ToolMetadata(name=name, description=name) for name in ("first", "backup")]
    for selector in (llamaindex.VeyaSingleSelector, llamaindex.VeyaMultiSelector):
        with pytest.raises(VeyaLowConfidenceError):
            selector(agent=runner, min_confidence=0.8).select(choices, "hello")
    fallback = llamaindex.VeyaSingleSelector(agent=runner, min_confidence=0.8, fallback_index=1)
    assert fallback.select(choices, "hello").selections[0].index == 1

    def unexpected_dispatch(*args):
        pytest.fail("the first engine was not configured as a fallback")

    engines = {"first": unexpected_dispatch, "backup": lambda state: ("fallback", state)}
    with pytest.raises(VeyaLowConfidenceError):
        llamaindex.VeyaQueryRouter(engines, agent=runner, min_confidence=0.8).query("hello")
    safe = llamaindex.VeyaQueryRouter(engines, fallback_key="backup", agent=runner, min_confidence=0.8)
    assert safe.query("hello") == ("fallback", "hello")


@pytest.mark.parametrize("index", [True, -1, 2, 1.5])
def test_agno_checks_fallback_index_before_inference(index):
    runner = SimpleNamespace(predict=lambda *args, **kwargs: pytest.fail("inference must not run"))
    adapter = agno.VeyaAgnoRouter(agent=runner, fallback_agent_index=index)
    with pytest.raises(ValueError, match="fallback_agent_index"):
        adapter.route("hello", ["first", "second"])


def test_routing_controls_refuse_positional_only_runners():
    class Runner:
        def predict(self, state, questions, task=None, /):
            pass

    with pytest.raises(ValueError, match="task"):
        require_router_controls(Runner(), "predict", {"task": "english"})


@pytest.mark.parametrize("answer", [None, {}, {"type": "noul"}, {"type": "noul", "noul": -1},
                                    {"type": "noul", "noul": float("nan")}])
def test_agno_guardrail_refuses_incomplete_or_invalid_answers(answer):
    runner = SimpleNamespace(predict=lambda *args, **kwargs: {"answers": {} if answer is None else {"q": answer}})
    guard = agno.VeyaAgnoGuardrail(agent=runner, questions={"q": {"type": "noul", "instructions": "Unsafe?"}})
    with pytest.raises(RuntimeError, match="Malformed decision"):
        guard.screen("hello")


@pytest.mark.parametrize("probabilities", [{"3": 1.0}, {"0": 0.0, "2": 0.0, "3": 1.0},
    {"0": 0.0, "1": 0.0, "2": 0.0, "4": 1.0},
    {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0},
    {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}, None])
def test_agno_guardrail_refuses_invalid_score_distributions(probabilities):
    answer = {"type": "score", "score": 3.0, "probabilities": probabilities}
    runner = SimpleNamespace(predict=lambda *args, **kwargs: {"answers": {"q": answer}})
    questions = {"q": {"type": "score", "instructions": "Harm?",
                       "criteria": ["none", "minor", "serious", "severe"]}}
    guard = agno.VeyaAgnoGuardrail(agent=runner, questions=questions)
    with pytest.raises(RuntimeError, match="score probabilities|'probabilities'"):
        guard.screen("hello")


def test_agno_guardrail_keeps_score_fallback_and_complete_distributions():
    questions = {"q": {"type": "score", "instructions": "Harm?",
                       "criteria": ["none", "minor", "serious", "severe"]}}
    for answer in ({"type": "score", "score": 3.0},
                   {"type": "score", "score": 3.0,
                    "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}}):
        runner = SimpleNamespace(predict=lambda *args, **kwargs: {"answers": {"q": answer}})
        with pytest.raises(agno.VeyaAgnoGuardrailError):
            agno.VeyaAgnoGuardrail(agent=runner, questions=questions).screen("hello")


def test_policy_depth_error_names_the_cli_option(tmp_path, monkeypatch):
    from veya import _eval_policy

    path = tmp_path / "policy.json"
    path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
    def refuse_depth(handle):
        raise RecursionError("excessive JSON nesting")
    monkeypatch.setattr(_eval_policy.json, "load", refuse_depth)
    with pytest.raises(EvalError, match="--gate-policy"):
        load_policy(str(path))


def test_policy_accepts_a_bom(tmp_path):
    policy = {"version": 1, "rules": [{"slice": {"model": "english"}, "metric": "choice_accuracy",
                                        "min_count": 1, "min": 0.5}]}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8-sig")
    assert load_policy(str(path)) == policy
