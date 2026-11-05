"""Corrected policies cannot replay decisions approved under canonical inference."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from laya import DecisionCache, Router
from laya.checkpoint import _checkpoint_signature
from laya.consistency import _KEY_VERSION, _canonical, _fingerprint
from laya.errors import DecisionCacheError
from laya.hooks import PredictContext, clear_default_hooks
from laya.replay import load_manifest


ROOT = Path(__file__).resolve().parents[1]
POLICY = "context-balanced-v1"
QUESTIONS = {"keep": {"type": "noul", "instructions": "Keep this?"}}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("LAYA_"):
            monkeypatch.delenv(name)
    clear_default_hooks()
    yield
    clear_default_hooks()


def predictor(directory, policy="canonical"):
    from laya.agent import Agent

    checkpoint = directory / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    cfg = {"max_len": 512, "head_max_len": 192}
    if policy != "canonical":
        cfg["decision_policy"] = policy
    (checkpoint / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (checkpoint / "model.safetensors").write_text("{}", encoding="utf-8")
    agent = object.__new__(Agent)
    agent.model_id, agent.subfolder, agent.revision = str(checkpoint), None, None
    agent.cfg, agent.decision_policy = cfg, policy
    agent.temperature, agent.temperature_by_options = [1, 1, 1], {}
    agent.lang_temperatures, agent.binning_map, agent.hooks = {}, None, []
    agent._checkpoint_signature = _checkpoint_signature(str(checkpoint), str(checkpoint / "model.safetensors"), {})
    agent.calls = 0

    def answer(state, questions, **kwargs):
        agent.calls += 1
        value = 0.51 if agent.decision_policy == "canonical" else 0.61
        return {"answers": {qid: {"type": "noul", "noul": value, "answer_confidence": value}
                            for qid in questions}, "usage": {"input_tokens": 1, "output_tokens": 0}}

    agent.system_one = answer
    agent.predict_batch = lambda states, questions, **kwargs: [answer(state, questions, **kwargs) for state in states]
    return agent, checkpoint


def configured(directory, policy, fingerprint):
    agent, checkpoint = predictor(directory, policy)
    cache = DecisionCache(str(directory / "decisions.sqlite"), ttl=None, maxsize=None,
                          fingerprint=fingerprint, required=True)
    router = Router(models={"english": str(checkpoint)}, hooks=[cache], performance_mode="default")
    router._agents["english"] = agent
    return router, cache, agent, checkpoint


@pytest.mark.parametrize("fingerprint", [None, "reviewed-policy"])
def test_policy_changes_separate_retained_answers_without_discarding_legacy_rows(tmp_path, fingerprint):
    router, cache, agent, _ = configured(tmp_path, "canonical", fingerprint)
    with cache:
        first = router.predict("hello", QUESTIONS, model="english")
        canonical_fingerprint = copy.deepcopy(_fingerprint(agent))
        agent.decision_policy = POLICY
        corrected = router.predict("hello", QUESTIONS, model="english")
        assert corrected["answers"]["keep"]["noul"] == 0.61
        assert _fingerprint(agent) != canonical_fingerprint
        agent.decision_policy = "canonical"
        assert router.predict("hello", QUESTIONS, model="english")["answers"] == first["answers"]
        assert agent.calls == 2
        assert cache.cache_info()["hits"] == 1
        assert cache.cache_info()["misses"] == 2


@pytest.mark.parametrize("fingerprint", [None, "reviewed-policy"])
def test_canonical_cache_layout_is_byte_compatible_with_existing_keys(tmp_path, fingerprint):
    agent, _ = predictor(tmp_path)
    cache = DecisionCache(fingerprint=fingerprint)
    with cache:
        context = PredictContext(states=["hello"], questions=QUESTIONS, model="english", agent=agent)
        identity = _fingerprint(agent)
        assert len(identity) == 14
        prefix = [_KEY_VERSION, fingerprint if fingerprint is not None else identity,
                  "english", None, None, None, QUESTIONS]
        digest = hashlib.blake2b(_canonical(prefix), digest_size=16)
        digest.update(b"\x00")
        digest.update(_canonical("hello"))
        assert cache._keys(context, context.states, False) == [digest.digest()]
        del agent.decision_policy
        assert cache._keys(context, context.states, False) == [digest.digest()]


def test_cleared_effective_policy_matches_inference_and_invalidates_balanced_approval(tmp_path):
    from laya.decision_policy import resolve_decision_policy

    router, cache, agent, checkpoint = configured(tmp_path, POLICY, "reviewed-policy")
    with cache:
        router.predict("hello", QUESTIONS, model="english")
        manifest = tmp_path / "replay.json"
        router.export_replay_manifest(manifest)
        approved = _fingerprint(agent)
        cold = Router(models={"english": str(checkpoint)}, hooks=[cache],
                      replay_manifest=str(manifest), performance_mode="default")
        cold._agents["english"] = agent
        agent.decision_policy = None
        assert resolve_decision_policy(agent.decision_policy) == "canonical"
        assert _fingerprint(agent) == approved[:-1]
        with pytest.raises(DecisionCacheError, match="metadata changed"):
            cold.try_replay("hello", QUESTIONS, model="english")


def test_required_cache_refuses_policy_mutation_during_prediction(tmp_path):
    router, cache, agent, _ = configured(tmp_path, "canonical", "reviewed-policy")
    original = agent.system_one

    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        agent.decision_policy = POLICY
        return value

    agent.system_one = mutate
    with cache, pytest.raises(DecisionCacheError, match="identity changed"):
        router.predict("hello", QUESTIONS, model="english")


@pytest.mark.parametrize("policy", ["canonical", POLICY])
def test_replay_preserves_effective_policy_and_refuses_resident_changes(tmp_path, policy):
    router, cache, agent, checkpoint = configured(tmp_path, policy, "reviewed-policy")
    with cache:
        expected = router.predict("hello", QUESTIONS, model="english")
        manifest = tmp_path / "replay.json"
        router.export_replay_manifest(manifest)
        approved = load_manifest(manifest)
        attributes = approved["models"]["english"]["attributes"]
        assert ("decision_policy" in attributes) == (policy != "canonical")
        cold = Router(models={"english": str(checkpoint)}, hooks=[cache],
                      replay_manifest=str(manifest), performance_mode="default")
        assert cold.validate_replay_manifest()
        assert cold.predict("hello", QUESTIONS, model="english")["answers"] == expected["answers"]
        assert not cold.loaded
        cold._agents["english"] = agent
        agent.decision_policy = POLICY if policy == "canonical" else "canonical"
        with pytest.raises(DecisionCacheError, match="metadata changed"):
            cold.validate_replay_manifest()
        with pytest.raises(DecisionCacheError, match="metadata changed"):
            cold.try_replay("hello", QUESTIONS, model="english")


def test_noncanonical_cold_replay_uses_policy_namespace_without_ml_imports(tmp_path):
    router, cache, _, checkpoint = configured(tmp_path, POLICY, "reviewed-policy")
    with cache:
        router.predict("hello", QUESTIONS, model="english")
        manifest = tmp_path / "replay.json"
        router.export_replay_manifest(manifest)
        store_id = cache.store_id
    code = """
import importlib.abc, sys
attempted=[]
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in ('torch','transformers','onnxruntime','tokenizers','onnx'):
   attempted.append(name)
   raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from laya import Router,DecisionCache
with DecisionCache(sys.argv[1],required=True,ttl=None,maxsize=None,
                   fingerprint='reviewed-policy',expected_store_id=sys.argv[4]) as cache:
 router=Router(models={'english':sys.argv[2]},hooks=[cache],replay_manifest=sys.argv[3],performance_mode='default')
 assert router.validate_replay_manifest()
 result=router.predict('hello',{'keep':{'type':'noul','instructions':'Keep this?'}},model='english')
 assert result['answers']['keep']['noul']==0.61,result
 assert not router.loaded
 assert not attempted,attempted
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "decisions.sqlite"),
                             str(checkpoint), str(manifest), store_id], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("configured_policy,override", [("canonical", POLICY), (POLICY, "canonical")])
def test_explicit_constructor_override_is_the_approved_effective_policy(tmp_path, configured_policy, override):
    router, cache, agent, checkpoint = configured(tmp_path, configured_policy, "reviewed-policy")
    agent.decision_policy = override
    router.agent_kwargs = {"decision_policy": override}
    with cache:
        expected = router.predict("hello", QUESTIONS, model="english")
        manifest = tmp_path / "replay.json"
        router.export_replay_manifest(manifest)
        cold = Router(models={"english": str(checkpoint)}, hooks=[cache], replay_manifest=str(manifest),
                      agent_kwargs={"decision_policy": override}, performance_mode="default")
        assert cold.validate_replay_manifest()
        assert cold.predict("hello", QUESTIONS, model="english")["answers"] == expected["answers"]
        other = "canonical" if override != "canonical" else POLICY
        with pytest.raises(DecisionCacheError, match="agent options differ"):
            Router(models={"english": str(checkpoint)}, hooks=[cache], replay_manifest=str(manifest),
                   agent_kwargs={"decision_policy": other}, performance_mode="default")
        missing = Router(models={"english": str(checkpoint)}, hooks=[cache],
                         replay_manifest=str(manifest), performance_mode="default")
        with pytest.raises(DecisionCacheError, match="policy configuration"):
            missing.validate_replay_manifest()


@pytest.mark.parametrize("raw,expected", [(None, {}), (" \t", {}),
                                         ("canonical", {"agent_kwargs": {"decision_policy": "canonical"}}),
                                         (" context-balanced-v1 ", {"agent_kwargs": {"decision_policy": POLICY}})])
def test_service_policy_environment_is_opt_in_and_shared(monkeypatch, raw, expected):
    from laya import serve
    if raw is not None:
        monkeypatch.setenv("LAYA_DECISION_POLICY", raw)
    assert serve._decision_policy_option() == expected


def test_invalid_http_policy_fails_before_runtime_or_router_construction(monkeypatch):
    import laya.router
    from laya import serve

    monkeypatch.setenv("LAYA_DECISION_POLICY", "automatic")
    monkeypatch.setattr(serve, "_router_runtime_setup", lambda *_: pytest.fail("runtime was constructed"))
    monkeypatch.setattr(laya.router, "Router", lambda **_: pytest.fail("router was constructed"))
    with pytest.raises(ValueError, match="LAYA_DECISION_POLICY"):
        serve.build_router()


def test_http_and_mcp_forward_the_same_local_policy(monkeypatch):
    import laya
    import laya.router
    from laya import serve

    monkeypatch.setenv("LAYA_PRELOAD", "0")
    monkeypatch.setenv("LAYA_DECISION_POLICY", POLICY)

    def build(**options):
        return SimpleNamespace(options=options, loaded=[], max_loaded=2)

    with monkeypatch.context() as http:
        http.setattr(laya.router, "Router", build)
        http.setattr(serve, "_router_runtime_setup", lambda *_: (None, None))
        http.setattr(serve, "_configure_decision_cache", lambda _: None)
        assert serve.build_router().options["agent_kwargs"] == {"decision_policy": POLICY}
    pytest.importorskip("mcp")
    from laya.mcp import server

    monkeypatch.setattr(laya, "Router", build)
    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setattr(server, "_PERFORMANCE_MODE", None)
    monkeypatch.setattr(server, "_BACKEND", None)
    monkeypatch.setattr(server, "_router_runtime_setup", lambda *_: (None, None))
    monkeypatch.setattr(server, "_configure_decision_cache", lambda _: None)
    assert server._ensure_router().options["agent_kwargs"] == {"decision_policy": POLICY}


def test_invalid_mcp_policy_stays_unconfigured_without_loading_models(monkeypatch):
    pytest.importorskip("mcp")
    from laya.mcp import server
    from laya.mcp.tools import ToolError

    monkeypatch.setenv("LAYA_DECISION_POLICY", "automatic")
    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setattr(server, "_router_runtime_setup", lambda *_: pytest.fail("runtime was constructed"))
    with pytest.raises(ToolError, match="LAYA_DECISION_POLICY"):
        server._ensure_router()
    assert server._ROUTER is None


@pytest.mark.parametrize("path,microbatch,width,budget", [
    ("/v1/systemone", False, None, 1024),
    ("/v1/systemone", True, None, 1024),
    ("/v1/systemone/batch", False, 256, 512),
    ("/v1/systemone", False, None, 256),
])
def test_http_policy_caps_each_real_and_probe_forward(tmp_path, monkeypatch, path, microbatch, width, budget):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from laya import serve
    from laya.decision_policy import CONTENT_FREE_PROBES

    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", str(budget))
    if microbatch:
        monkeypatch.setenv("LAYA_MICROBATCH_MAX_STATES", "2")
        monkeypatch.setenv("LAYA_MICROBATCH_WAIT_MS", "0")
    agent, checkpoint = predictor(tmp_path, POLICY)
    # Exercise the real Agent entry point and policy splitter with a controlled
    # backend, recording each canonical forward's padded token ceiling.
    del agent.system_one, agent.predict_batch
    agent.performance_config = None
    agent.hooks_raise, agent.hooks_timeout, agent._hooks_lock = True, None, None
    forwards = []

    def infer(states, questions, *, batch_size=None, max_len=None, full_precision=False, **_):
        assert full_precision
        size = batch_size or len(states)
        limit = max_len if max_len is not None else agent.cfg["max_len"]
        for start in range(0, len(states), size):
            group = states[start:start + size]
            forwards.append((len(group) * len(questions), limit,
                             all(state in CONTENT_FREE_PROBES for state in group)))
        return [{"model": "fixture", "answers": {
            qid: {"_decision_probabilities": [0.6, 0.3, 0.1], "action": {"act_probability": 1.0}}
            for qid in questions}, "usage": {"input_tokens": len(questions) * 20}}
                for _ in states]

    agent._infer_batch = infer
    router = Router(models={"english": str(checkpoint)}, performance_mode="default")
    router._agents["english"] = agent
    questions = {"choice": {"type": "choice", "instructions": "Choose a category.",
                            "criteria": {"a": "First", "b": "Second", "c": "Third"}}}
    states = ["hello", "again"] if path.endswith("/batch") else ["hello"]
    body = {"questions": questions, "model": "english"}
    if path.endswith("/batch"):
        body.update(states=states, batch_size=64)
    else:
        body["state"] = states[0]
    if width is not None:
        body["max_len"] = width
    with TestClient(serve.create_app(router=router)) as client:
        response = client.post(path, json=body)
    if (width if width is not None else agent.cfg["max_len"]) > budget:
        assert response.status_code == 422, response.text
        assert not forwards
    else:
        assert response.status_code == 200, response.text
        assert sum(rows for rows, _, probe in forwards if probe) == len(CONTENT_FREE_PROBES) * 3
        assert sum(rows for rows, _, probe in forwards if not probe) == len(states) * 3
        assert all(rows * limit <= budget for rows, limit, _ in forwards)
