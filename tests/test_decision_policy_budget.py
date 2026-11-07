"""Optional contextual decisions cannot bypass Code Mode's forward-work limits."""
from types import SimpleNamespace

import pytest

from veya.mcp._code_budget import check_operation_budget
from veya.mcp.tools import ToolError


STATE = {"message": "Check this request"}
QUESTIONS = {"team": {"type": "choice", "instructions": "Pick a team",
                       "criteria": {"alpha": "First", "beta": "Second", "other": "Other"}}}


class Tokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3

    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))
        return {"input_ids": ids[:kwargs["max_length"]] if kwargs.get("truncation") else ids}


class Agent:
    def __init__(self, policy="context-balanced-v1"):
        self.cfg = {"max_len": 512, "head_max_len": 192}
        self.decision_policy = policy
        self.tok = Tokenizer()

    @staticmethod
    def _to_internal(question):
        return {"t": question["type"], "ins": question["instructions"], "crit": question.get("criteria")}


class Router:
    def __init__(self, agent=None):
        self._agents = {"english": agent} if agent is not None else {}

    def load(self, *args, **kwargs):
        raise AssertionError("budget checking must not load a checkpoint")


def payload(**kwargs):
    return dict({"state": STATE, "questions": QUESTIONS, "model": "english"}, **kwargs)


def check(method="predict", arguments=None, *, agent=None, maximum=512, **kwargs):
    return check_operation_budget(method, payload() if arguments is None else arguments,
                                  Router(agent if agent is not None else Agent()),
                                  SimpleNamespace(max_batch_tokens=maximum), **kwargs)


@pytest.mark.parametrize("method", ["predict", "shortlist", "tournament", "predict_long"])
def test_rotation_and_probe_work_exceeds_a_canonical_only_budget(method):
    check(method, agent=Agent("canonical"))
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check(method)
    check(method, maximum=12 * 512)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check(method, maximum=12 * 512 - 1)


@pytest.mark.parametrize("kind,criteria,count", [
    ("noul", None, 2), ("score", ["low", "medium", "high"], 3),
])
def test_each_question_type_accounts_for_its_rotations_and_fixed_probes(kind, criteria, count):
    question = {"type": kind, "instructions": "Decide"}
    if criteria is not None:
        question["criteria"] = criteria
    arguments = payload(questions={"q": question})
    check(arguments=arguments, maximum=4 * count * 512)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check(arguments=arguments, maximum=4 * count * 512 - 1)


@pytest.mark.parametrize("method", ["decide", "preset"])
def test_generated_questions_use_the_same_policy_budget(method):
    arguments = payload()
    if method == "decide":
        arguments["schema"] = {"type": "object", "properties": {"ready": {"type": "boolean"}}}
    else:
        arguments["preset"] = "guard"
    check(method, arguments, agent=Agent("canonical"), maximum=65536)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check(method, arguments)


def test_batch_sums_policy_cost_and_only_authorized_replay_skips_work():
    arguments = {"requests": [payload(), payload()]}
    check("predict_batch", arguments, maximum=24 * 512)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check("predict_batch", arguments, maximum=12 * 512)
    check("predict_batch", arguments, maximum=12 * 512, replayed=(0,))
    check("predict_batch", arguments, maximum=1, replayed=(0, 1))
    arguments["requests"][0]["state"] = {"text": "x" * 50001}
    with pytest.raises(ToolError, match="max_state_chars"):
        check("predict_batch", arguments, maximum=1, replayed=(0, 1))


def test_cold_preflight_defers_actual_policy_but_postload_check_cannot_bypass_it():
    router = Router()
    limits = SimpleNamespace(max_batch_tokens=512)
    assert check_operation_budget("predict", payload(), router, limits, preflight=True)
    router._agents["english"] = Agent()
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check_operation_budget("predict", payload(), router, limits)


def test_local_batch_eviction_cannot_discount_an_unknown_checkpoint_policy(monkeypatch):
    from veya.router import Router as LocalRouter

    router = LocalRouter(models={"english": "english-fixture", "multilingual": "multi-fixture"},
                         backend="onnx", max_loaded=1, preload=False, performance_mode="default")
    monkeypatch.setattr(router, "_build", lambda name: Agent() if name == "english" else Agent("canonical"))
    arguments = {"requests": [payload(max_len=512), payload(model="multilingual", max_len=512)]}
    limits = SimpleNamespace(max_batch_tokens=1024)
    assert check_operation_budget("predict_batch", arguments, router, limits, preflight=True)
    # The same load order as Code Mode: the second checkpoint evicts the first.
    assert router.load("english").decision_policy == "context-balanced-v1"
    assert router.load("multilingual").decision_policy == "canonical"
    assert router.loaded == ["multilingual"]
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check_operation_budget("predict_batch", arguments, router, limits)
    router.agent_kwargs["decision_policy"] = "canonical"
    check_operation_budget("predict_batch", arguments, router, limits)


def test_checkpoint_default_is_respected_but_explicit_canonical_takes_precedence():
    agent = Agent()
    agent.cfg["decision_policy"] = "context-balanced-v1"
    del agent.decision_policy
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check(agent=agent)
    agent.decision_policy = "canonical"
    check(agent=agent)


@pytest.mark.parametrize("policy", ["unknown-policy", "", 1, False, []])
def test_unknown_resident_policy_is_not_assumed_to_have_canonical_cost(policy):
    with pytest.raises(ToolError, match="decision_policy"):
        check(agent=Agent(policy), maximum=1000000)


def test_shortlist_counts_reduced_options_and_unscaled_embedding_work():
    criteria = {"label%d" % index: "description" for index in range(200)}
    arguments = payload(questions={"q": {"type": "choice", "instructions": "Pick", "criteria": criteria}}, k=8)
    maximum = (4 * 8 + 201) * 512
    check("shortlist", arguments, maximum=maximum)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check("shortlist", arguments, maximum=maximum - 1)


def test_tournament_counts_each_round_at_its_bounded_group_width():
    criteria = {"label%d" % index: "description" for index in range(200)}
    arguments = payload(questions={"q": {"type": "choice", "instructions": "Pick", "criteria": criteria}},
                        group_size=8)
    # 25 groups, then four groups, then the final question; each uses at most eight options.
    maximum = 30 * 4 * 8 * 512
    check("tournament", arguments, maximum=maximum)
    with pytest.raises(ToolError, match="max_batch_tokens"):
        check("tournament", arguments, maximum=maximum - 1)


def test_routing_only_still_has_no_neural_cost():
    check("route", maximum=1)
