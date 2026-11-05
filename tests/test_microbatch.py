"""Opt-in cross-request collection: resource bounds, cancellation and HTTP semantics."""
import asyncio
import builtins
import threading
from types import MethodType

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from laya import DecisionCache, Router
from laya._microbatch import Microbatcher, MicrobatchUnavailable
from laya.errors import DecisionCacheError
from laya.onnx_agent import ONNXAgent, _NativeONNXAgent
from laya import serve


QUESTIONS = {"q": {"type": "noul", "instructions": "Does this need attention?"}}


class CountingAgent:
    cfg = {"max_len": 512, "head_max_len": 192}
    model_id = "microbatch-fixture"
    revision = "fixture-v1"
    hooks = ()

    def __init__(self):
        self.singles = []
        self.batches = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.required_error = False

    def result(self, state):
        if state == "block":
            self.entered.set()
            assert self.release.wait(5), "test failed to release model"
        if state == "bad":
            if self.required_error:
                raise DecisionCacheError("private store location")
            raise ValueError("only this bad state is invalid")
        return {"model": "fixture", "answers": {"q": {"type": "noul", "noul": 0.51,
                "answer_confidence": 0.51, "state": state}},
                "usage": {"input_tokens": 10, "output_tokens": 0}}

    def system_one(self, state, questions, **kwargs):
        self.singles.append(state)
        return self.result(state)

    def predict_batch(self, states, questions, batch_size=None, **kwargs):
        outputs = []
        for start in range(0, len(states), batch_size or len(states)):
            chunk = states[start:start + (batch_size or len(states))]
            self.batches.append(list(chunk))
            outputs.extend(self.result(state) for state in chunk)
        return outputs


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in ("LAYA_API_KEY", "LAYA_TOURNAMENT_GROUP", "LAYA_JEV_STRICT", "LAYA_IDLE_UNLOAD_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYA_MICROBATCH_MAX_STATES", "8")
    monkeypatch.setenv("LAYA_MICROBATCH_WAIT_MS", "10")


def setup_router(hooks=None, agent_type=ONNXAgent):
    # A real built-in type with controlled methods avoids loading any checkpoint.
    agent = object.__new__(agent_type)
    agent.__dict__.update(vars(CountingAgent()))
    for name in ("cfg", "model_id", "revision"):
        setattr(agent, name, getattr(CountingAgent, name))
    for name in ("result", "system_one", "predict_batch"):
        setattr(agent, name, MethodType(getattr(CountingAgent, name), agent))
    router = Router(hooks=hooks)
    router.attach("english", agent)
    return router, agent


def request(state, **controls):
    return {"state": state, "questions": QUESTIONS, "model": "english", **controls}


async def send(app, payloads, path="/v1/systemone"):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            return await asyncio.gather(*(client.post(path, json=payload) for payload in payloads))


@pytest.mark.parametrize("agent_type", [ONNXAgent, _NativeONNXAgent])
def test_compatible_requests_share_forward_pass_and_preserve_order(agent_type):
    router, agent = setup_router(agent_type=agent_type)
    responses = asyncio.run(send(serve.create_app(router), [request(str(i)) for i in range(8)]))
    assert [r.status_code for r in responses] == [200] * 8
    assert [r.json()["answers"]["q"]["state"] for r in responses] == [str(i) for i in range(8)]
    assert agent.batches == [[str(i) for i in range(8)]]
    assert agent.singles == []
    assert all("queue;dur=" in r.headers["Server-Timing"] for r in responses)


def test_native_onnx_planning_never_imports_torch(monkeypatch):
    router, _ = setup_router(agent_type=_NativeONNXAgent)
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        assert name != "torch" and not name.startswith("torch."), "ONNX planning imported Torch"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    payloads = [(str(i), QUESTIONS, "english", {}) for i in range(2)]
    assert serve._microbatch_plan(router, payloads, 1024) == 2


def test_default_off_preserves_single_calls_and_timing_headers(monkeypatch):
    monkeypatch.delenv("LAYA_MICROBATCH_MAX_STATES")
    router, agent = setup_router()
    responses = asyncio.run(send(serve.create_app(router), [request("one"), request("two")]))
    assert agent.singles == ["one", "two"]
    assert not agent.batches
    assert all("queue;dur=" not in r.headers["Server-Timing"] for r in responses)


def test_aggregate_budget_uses_actual_resident_width(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", "2048")
    router, agent = setup_router()
    agent.cfg = {"max_len": 1024}
    responses = asyncio.run(send(serve.create_app(router), [request(str(i)) for i in range(6)]))
    assert all(r.status_code == 200 for r in responses)
    assert [len(batch) for batch in agent.batches] == [2, 2, 2]


def test_unknown_width_and_custom_hooks_stay_serial():
    class Observer:
        def __init__(self):
            self.starts = 0

        def on_predict_start(self, ctx):
            self.starts += 1
    observer = Observer()
    router, agent = setup_router([observer])
    responses = asyncio.run(send(serve.create_app(router), [request("one"), request("two")]))
    assert all(r.status_code == 200 for r in responses)
    assert observer.starts == 2
    assert agent.singles == ["one", "two"]
    router, agent = setup_router()
    agent.cfg = {}
    asyncio.run(send(serve.create_app(router), [request("one"), request("two")]))
    assert agent.singles == ["one", "two"]


def test_control_differences_do_not_share_confidence_policy():
    router, agent = setup_router()
    responses = asyncio.run(send(serve.create_app(router), [request("one"), request("two", min_confidence=0.9)]))
    assert "abstention" not in responses[0].json()["answers"]["q"]
    assert responses[1].json()["answers"]["q"]["abstention"] == "abstained"
    assert agent.singles == ["one", "two"]


def test_custom_agent_retains_language_control_on_individual_path():
    class LanguageAgent(CountingAgent):
        def system_one(self, state, questions, lang=None, **kwargs):
            self.singles.append(lang)
            return self.result(str(lang))

    agent = LanguageAgent()
    router = Router()
    router.attach("english", agent)
    responses = asyncio.run(send(serve.create_app(router), [request("one", lang="fr"), request("two", lang="fr")]))
    assert [r.json()["answers"]["q"]["state"] for r in responses] == ["fr", "fr"]
    assert agent.singles == ["fr", "fr"]
    assert agent.batches == []


def test_bad_request_does_not_fail_or_leak_error_to_neighbours():
    router, agent = setup_router()
    responses = asyncio.run(send(serve.create_app(router), [request("one"), request("bad"), request("three")]))
    assert [r.status_code for r in responses] == [200, 422, 200]
    assert "bad state" in responses[1].text
    assert "bad state" not in responses[0].text + responses[2].text
    assert agent.singles == ["one", "bad", "three"]


def test_required_cache_failure_is_terminal_without_individual_retry():
    router, agent = setup_router()
    agent.required_error = True
    payloads = [(str(i), QUESTIONS, "english", {}) for i in range(2)]
    payloads[1] = ("bad", QUESTIONS, "english", {})
    with pytest.raises(DecisionCacheError):
        serve._run_microbatch(router, payloads, 131072)
    assert agent.singles == []


def test_batch_endpoint_remains_explicit_batch_path():
    router, agent = setup_router()
    responses = asyncio.run(send(serve.create_app(router), [
        {"states": ["one", "two"], "questions": QUESTIONS, "model": "english"}], "/v1/systemone/batch"))
    assert responses[0].status_code == 200
    assert agent.batches == [["one", "two"]]
    assert "queue;dur=" not in responses[0].headers["Server-Timing"]


def test_duplicate_and_later_repeat_decisions_use_first_stored_value():
    cache = DecisionCache(maxsize=None, fingerprint="microbatch-v1")
    router, agent = setup_router([cache])
    app = serve.create_app(router)
    responses = asyncio.run(send(app, [request("repeat")] * 4))
    first = [r.json()["answers"] for r in responses]
    assert all(result == first[0] for result in first)
    assert agent.singles == ["repeat"]
    assert agent.batches == []
    later = asyncio.run(send(app, [request("repeat")] * 4))
    assert all(r.json()["answers"] == first[0] for r in later)
    assert agent.singles == ["repeat"]
    assert agent.batches == []


def test_distinct_cached_states_can_still_share_forward_pass():
    cache = DecisionCache(maxsize=None, fingerprint="microbatch-v1")
    router, agent = setup_router([cache])
    responses = asyncio.run(send(serve.create_app(router), [request("one"), request("two")]))
    assert all(r.status_code == 200 for r in responses)
    assert agent.batches == [["one", "two"]]
    assert agent.singles == []


def test_queue_stays_bounded_when_waiting_callers_cancel():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def execute(jobs):
            seen.extend(job.payload for job in jobs)
            entered.set()
            await release.wait()
            return [(job.payload, None, 1.0, 0.0) for job in jobs]

        batcher = Microbatcher(execute, capacity=1, max_states=2, wait_seconds=0)
        first = asyncio.create_task(batcher.submit("same", "first"))
        await entered.wait()
        queued = asyncio.create_task(batcher.submit("same", "cancelled"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        with pytest.raises(MicrobatchUnavailable):
            await batcher.submit("same", "overflow")
        release.set()
        assert (await first)[0] == "first"
        await batcher.close()
        assert seen == ["first"]
    asyncio.run(run())


def test_shutdown_settles_active_and_queued_futures():
    async def run():
        entered = asyncio.Event()

        async def execute(jobs):
            entered.set()
            await asyncio.Event().wait()

        batcher = Microbatcher(execute, capacity=2, max_states=2, wait_seconds=0)
        first = asyncio.create_task(batcher.submit("a", 1))
        await entered.wait()
        queued = asyncio.create_task(batcher.submit("a", 2))
        await asyncio.sleep(0)
        await batcher.close()
        for future in (first, queued):
            with pytest.raises(MicrobatchUnavailable):
                await future
    asyncio.run(run())


def test_cancelled_job_waiting_for_legacy_gate_never_reaches_model():
    router, agent = setup_router()
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                blocked = asyncio.create_task(client.post("/v1/systemone/batch", json={
                    "states": ["block"], "questions": QUESTIONS, "model": "english"}))
                for _ in range(1000):
                    if agent.entered.is_set():
                        break
                    await asyncio.sleep(0.001)
                assert agent.entered.is_set()
                cancelled = asyncio.create_task(client.post("/v1/systemone", json=request("cancelled")))
                await asyncio.sleep(0.04)
                cancelled.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
                agent.release.set()
                assert (await blocked).status_code == 200
                assert (await client.post("/v1/systemone", json=request("after"))).status_code == 200
        assert "cancelled" not in agent.singles
    asyncio.run(run())


@pytest.mark.parametrize("name,value", [("LAYA_MICROBATCH_MAX_STATES", "1"),
    ("LAYA_MICROBATCH_MAX_STATES", "65"), ("LAYA_MICROBATCH_MAX_STATES", "oops"),
    ("LAYA_MICROBATCH_WAIT_MS", "nan"), ("LAYA_MICROBATCH_WAIT_MS", "inf"),
    ("LAYA_MICROBATCH_WAIT_MS", "-1")])
def test_invalid_deployment_config_fails_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    router, _ = setup_router()
    with pytest.raises(ValueError, match="LAYA_MICROBATCH"):
        serve.create_app(router)
