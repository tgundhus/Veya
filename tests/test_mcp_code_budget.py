"""Code-mode work limits must reject oversized requests before model work."""
import copy
from types import SimpleNamespace

import pytest

from laya.mcp._code_budget import check_operation_budget
from laya.mcp.tools import ToolError


QUESTIONS = {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}
STATE = {"message": "Please review the payment request"}


class Tokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3

    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))
        if kwargs.get("truncation"):
            ids = ids[:kwargs["max_length"]]
        return {"input_ids": ids}


class Agent:
    cfg = {"max_len": 512, "head_max_len": 192}
    tok = Tokenizer()

    def _to_internal(self, question):
        return {"t": question["type"], "ins": question["instructions"],
                "criteria": question.get("criteria")}

    def system_one(self, *args, **kwargs):
        raise AssertionError("budget validation must not run a model")


class Router:
    def __init__(self, resident=True):
        self._agents = {"english": Agent()} if resident else {}

    def load(self, *args, **kwargs):
        raise AssertionError("budget validation must not load a model")


def check(method, arguments=None, limits=None, resident=True):
    payload = {"state": STATE, "questions": QUESTIONS, "model": "english"}
    if arguments is not None:
        payload.update(arguments)
    check_operation_budget(method, payload, Router(resident), limits or SimpleNamespace())


def choices(count):
    return {"pick": {"type": "choice", "instructions": "Pick one", "criteria": {
        "label%d" % index: "description%d" % index for index in range(count)}}}


@pytest.mark.parametrize("method", ["predict", "route", "shortlist", "tournament", "predict_long"])
def test_regular_operations_validate_without_loading_or_forwarding(method):
    check(method)


@pytest.mark.parametrize("method", ["predict", "route", "shortlist", "tournament", "predict_long"])
def test_question_count_is_limited_for_every_question_operation(method):
    questions = {"q%d" % index: QUESTIONS["urgent"] for index in range(65)}
    with pytest.raises(ToolError, match="max_questions"):
        check(method, {"questions": questions})


@pytest.mark.parametrize("method", ["predict", "route", "shortlist", "tournament", "predict_long"])
def test_state_size_is_limited_before_any_operation(method):
    with pytest.raises(ToolError, match="max_state_chars"):
        check(method, {"state": {"text": "x" * 50001}})


@pytest.mark.parametrize("field", ["max_len", "head_max_len"])
@pytest.mark.parametrize("value", [8193, 1000000000000])
def test_explicit_token_budgets_cannot_exceed_host_limits(field, value):
    with pytest.raises(ToolError, match=field):
        check("predict", {field: value})


@pytest.mark.parametrize("method", ["predict_batch", "route_batch"])
def test_batch_cardinality_is_checked_before_nested_validation(method):
    with pytest.raises(ToolError, match="max_states"):
        check_operation_budget(method, {"requests": [None] * 65}, Router(), SimpleNamespace())


def test_batch_uses_sum_of_actual_resident_work_and_keeps_original_input():
    request = {"state": STATE, "questions": QUESTIONS, "model": "english", "max_len": 512}
    arguments = {"requests": [copy.deepcopy(request), copy.deepcopy(request)]}
    original = copy.deepcopy(arguments)
    check_operation_budget("predict_batch", arguments, Router(), SimpleNamespace(max_batch_tokens=1024))
    assert arguments == original
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check_operation_budget("predict_batch", arguments, Router(), SimpleNamespace(max_batch_tokens=1023))


def test_each_batch_override_is_checked():
    request = {"state": STATE, "questions": QUESTIONS, "model": "english", "max_len": 8193}
    with pytest.raises(ToolError, match="max_len"):
        check_operation_budget("predict_batch", {"requests": [request]}, Router(), SimpleNamespace())


def test_configured_resident_width_is_checked_without_loading():
    router = Router()
    router._agents["english"].cfg = {"max_len": 16384, "head_max_len": 192}
    with pytest.raises(ToolError, match="max_len"):
        check_operation_budget("predict", {"state": STATE, "questions": QUESTIONS, "model": "english"},
                               router, SimpleNamespace())


def test_unknown_cold_agent_is_bounded_conservatively():
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check("predict", limits=SimpleNamespace(max_batch_tokens=8191), resident=False)


def test_score_levels_and_total_option_work_are_bounded():
    with pytest.raises(ToolError, match="max_score_levels"):
        check("predict", {"questions": {"score": {"type": "score", "instructions": "Rate", "criteria": ["x"] * 33}}})
    questions = {"q%d" % i: choices(100)["pick"] for i in range(6)}
    with pytest.raises(ToolError, match="max_total_options"):
        check("predict", {"questions": questions})


def test_large_choices_only_allowed_for_bounded_reduction_operations():
    with pytest.raises(ToolError, match="max_choice_options"):
        check("predict", {"questions": choices(101)})
    check("shortlist", {"questions": choices(200)})
    check("tournament", {"questions": choices(200)})
    with pytest.raises(ToolError, match="max_tournament_options"):
        check("tournament", {"questions": choices(256)})


@pytest.mark.parametrize("method,argument,value", [("shortlist", "k", 101), ("tournament", "group_size", 101)])
def test_reduction_controls_cannot_make_oversized_final_questions(method, argument, value):
    with pytest.raises(ToolError, match=argument):
        check(method, {argument: value})


def test_shortlist_embedding_work_is_included_in_total_cost():
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check("shortlist", {"questions": choices(30)}, SimpleNamespace(max_batch_tokens=1024))
    with pytest.raises(ToolError, match="embedding width"):
        check("shortlist", {"questions": choices(30), "max_len": 256, "head_max_len": 192},
              SimpleNamespace(max_token_budget=256))


def test_tournament_generated_question_count_is_bounded():
    with pytest.raises(ToolError, match="round would exceed max_questions"):
        check("tournament", {"questions": choices(255), "group_size": 2})


def test_small_stride_cannot_expand_a_scan_into_hundreds_of_windows():
    with pytest.raises(ToolError, match="max_scan_windows"):
        check("predict_long", {"state": {"text": "word " * 200}, "window": 2, "stride": 1})
    with pytest.raises(ToolError, match="resident checkpoint tokenizer"):
        check("predict_long", resident=False)


def test_scan_work_includes_every_question_in_every_window():
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check("predict_long", {"state": {"text": "word " * 100}, "window": 20, "stride": 10},
              SimpleNamespace(max_batch_tokens=1000))


def test_decide_huge_ranges_are_refused_before_schema_planning(monkeypatch):
    from laya import structured
    monkeypatch.setattr(structured, "questions_from_json_schema",
                        lambda *args: pytest.fail("oversized schema reached allocation"))
    schema = {"type": "object", "properties": {"level": {
        "type": "integer", "minimum": 0, "maximum": 1000000000000}}}
    with pytest.raises(ToolError, match="max_score_levels"):
        check("decide", {"schema": schema})


def test_decide_local_reference_ranges_are_checked_before_planning(monkeypatch):
    from laya import structured
    monkeypatch.setattr(structured, "questions_from_json_schema",
                        lambda *args: pytest.fail("oversized referenced schema reached allocation"))
    schema = {"type": "object", "properties": {"level": {"$ref": "#/$defs/Level"}},
              "$defs": {"Level": {"type": "integer", "minimum": 0, "maximum": 1000000}}}
    with pytest.raises(ToolError, match="max_score_levels"):
        check("decide", {"schema": schema})


def test_normal_schema_and_builtin_preset_remain_supported():
    check("decide", {"schema": {"type": "object", "properties": {"keep": {"type": "boolean"}}}})
    check("preset", {"preset": "guard"})
    check_operation_budget("status", {}, None, SimpleNamespace())


def test_routing_only_does_not_charge_model_forward_budget():
    check("route", limits=SimpleNamespace(max_batch_tokens=1))


def test_preflight_exposes_bounded_cold_requests_then_full_check_refuses_unknown_scan():
    args = {"state": STATE, "questions": QUESTIONS, "model": "english", "window": 2, "stride": 1}
    original = copy.deepcopy(args)
    requests = check_operation_budget("predict_long", args, Router(False), SimpleNamespace(), preflight=True)
    assert requests[0]["state"] == STATE
    assert requests[0]["questions"] == QUESTIONS
    assert args == original
    with pytest.raises(ToolError, match="resident checkpoint tokenizer"):
        check_operation_budget("predict_long", args, Router(False), SimpleNamespace())


def test_preflight_defers_unknown_actual_width_but_still_checks_raw_limits():
    args = {"state": STATE, "questions": QUESTIONS, "model": "english"}
    assert check_operation_budget("predict", args, Router(False),
                                  SimpleNamespace(max_batch_tokens=512), preflight=True)
    with pytest.raises(ToolError, match="max_len"):
        check_operation_budget("predict", {**args, "max_len": 8193}, Router(False),
                               SimpleNamespace(), preflight=True)


def test_preset_preflight_returns_the_same_state_identity_as_the_preset_operation():
    args = {"state": {"message": "text to check"}, "preset": "guard"}
    original = copy.deepcopy(args)
    requests = check_operation_budget("preset", args, Router(), SimpleNamespace(), preflight=True)
    assert requests[0]["state"] == {"prompt": "text to check"}
    assert requests[0]["questions"]
    assert args == original


@pytest.mark.parametrize("method", ["predict", "decide", "preset"])
def test_verified_zero_forward_hits_skip_cost_but_still_validate_inputs(method):
    args = {"state": STATE, "model": "english"}
    if method == "decide":
        args["schema"] = {"type": "object", "properties": {"keep": {"type": "boolean"}}}
    elif method == "preset":
        args["preset"] = "guard"
        args.pop("model")
    else:
        args["questions"] = {"q%d" % index: QUESTIONS["urgent"] for index in range(64)}
    limits = SimpleNamespace(max_batch_tokens=1)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check_operation_budget(method, args, Router(False), limits)
    assert check_operation_budget(method, args, Router(False), limits, replayed=(0,))
    with pytest.raises(ToolError, match="max_state_chars"):
        check_operation_budget(method, {**args, "state": {"text": "x" * 50001}},
                               Router(False), limits, replayed=(0,))


def test_replayed_batch_indices_skip_only_their_authorized_work():
    args = {"requests": [{"state": STATE, "questions": QUESTIONS, "model": "english"},
                         {"state": STATE, "questions": QUESTIONS, "model": "english"}]}
    limits = SimpleNamespace(max_batch_tokens=512)
    assert check_operation_budget("predict_batch", args, Router(), limits, replayed={0})
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check_operation_budget("predict_batch", args, Router(), limits)
    args["requests"][0]["max_len"] = 8193
    with pytest.raises(ToolError, match="max_len"):
        check_operation_budget("predict_batch", args, Router(), limits, replayed={0})


@pytest.mark.parametrize("replayed", [(True,), (-1,), (1,), ("0",), "0", None])
def test_invalid_replay_indices_are_refused(replayed):
    args = {"state": STATE, "questions": QUESTIONS, "model": "english"}
    with pytest.raises(ToolError, match="request ind|unknown request"):
        check_operation_budget("predict", args, Router(), SimpleNamespace(), replayed=replayed)


@pytest.mark.parametrize("method", ["predict_long", "tournament", "shortlist", "route", "route_batch", "status"])
def test_replay_cannot_bypass_operations_with_other_work(method):
    with pytest.raises(ToolError, match="cannot skip work"):
        check_operation_budget(method, {}, Router(), SimpleNamespace(), replayed=(0,))
