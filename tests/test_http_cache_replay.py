"""Verified HTTP hits avoid model queues; misses retain bounded original execution."""
import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from laya import DecisionCache, Router
from laya import serve
from laya.errors import DecisionCacheError
from laya.onnx_agent import _NativeONNXAgent


QUESTIONS = {"q": {"type": "noul", "instructions": "Is this urgent?"}}


def result(state):
    return {"model": "fixture", "answers": {"q": {"type": "noul", "noul": 0.51,
            "answer_confidence": 0.51, "state": state}},
            "usage": {"input_tokens": 10, "output_tokens": 0}}


class Predictor:
    cfg = {"max_len": 512, "head_max_len": 192}
    model_id = "http-replay-fixture"
    revision = "v1"
    hooks = ()

    def system_one(self, state, questions, **kwargs):
        self.calls.append(state)
        if state == "block-model":
            self.entered.set()
            assert self.release.wait(5), "test did not release model work"
        return result(state)

    def predict_batch(self, states, questions, **kwargs):
        self.batches.append(list(states))
        return [self.system_one(state, questions, **kwargs) for state in states]


def configured_router(probe=None):
    agent = object.__new__(_NativeONNXAgent)
    agent.cfg, agent.model_id, agent.revision = Predictor.cfg, Predictor.model_id, Predictor.revision
    agent.calls, agent.batches = [], []
    agent.entered, agent.release = threading.Event(), threading.Event()
    agent.system_one = MethodType(Predictor.system_one, agent)
    agent.predict_batch = MethodType(Predictor.predict_batch, agent)
    router = Router()
    router.attach("english", agent)
    # Scheduler-focused tests supply the authoritative API outcome, without encoding
    # manifest internals here. End-to-end cases below exercise real exported metadata.
    router._replay_path = "configured-manifest"
    router.validate_replay_manifest = lambda: True
    router.try_replay = probe or (lambda state, questions, **kwargs:
                                  copy.deepcopy(result(state)) if state == "cached" else None)
    router.predict = lambda state, questions, **kwargs: agent.system_one(state, questions, **kwargs)
    router.predict_batch = lambda requests, **kwargs: agent.predict_batch(
        [request["state"] for request in requests], requests[0]["questions"], **kwargs)
    return router, agent


def payload(state="cached", **controls):
    return {"state": state, "questions": QUESTIONS, "model": "english", **controls}


async def wait_for(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(0.001)
    try:
        await asyncio.wait_for(poll(), 2)
    except asyncio.TimeoutError:
        raise AssertionError("worker did not reach the expected point") from None


async def drive(app, requests):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            return await asyncio.gather(*(client.post("/v1/systemone", json=request) for request in requests))


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in ("LAYA_API_KEY", "LAYA_TOURNAMENT_GROUP", "LAYA_JEV_STRICT", "LAYA_IDLE_UNLOAD_SECONDS",
                 "LAYA_CACHE", "LAYA_CACHE_PATH", "LAYA_CACHE_REQUIRED", "LAYA_CACHE_REQUIRE_EXISTING",
                 "LAYA_CACHE_EXPECTED_STORE_ID", "LAYA_REPLAY_MANIFEST", "LAYA_HTTP_REPLAY_FAST_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYA_MICROBATCH_MAX_STATES", "8")
    monkeypatch.setenv("LAYA_MICROBATCH_WAIT_MS", "10")
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "16")


def test_verified_hits_never_enter_collection_or_model(monkeypatch):
    from laya import _microbatch

    original = _microbatch.Microbatcher
    queue_sizes, threads = [], []

    class CacheOnly(original):
        def __init__(self, *args, **kwargs):
            queue_sizes.append(kwargs["max_states"])
            assert kwargs["wait_seconds"] == 0
            super().__init__(*args, **kwargs)

    def probe(state, questions, **kwargs):
        threads.append(threading.current_thread().name)
        return result(state)

    monkeypatch.setattr(_microbatch, "Microbatcher", CacheOnly)
    monkeypatch.setenv("LAYA_MICROBATCH_WAIT_MS", "1000")
    router, agent = configured_router(probe)
    responses = asyncio.run(drive(serve.create_app(router), [payload()] * 4))
    assert queue_sizes == [16]
    assert all(r.status_code == 200 and r.headers["X-Decision-Cache"] == "hit" for r in responses)
    assert all(r.headers["X-Inference-Time-Ms"] == "0.00" for r in responses)
    assert all(", cache;dur=" in r.headers["Server-Timing"] for r in responses)
    assert all(name.startswith("laya-replay") for name in threads)
    assert not agent.calls and not agent.batches


def test_misses_still_share_bounded_microbatch():
    router, agent = configured_router()
    responses = asyncio.run(drive(serve.create_app(router), [payload("one"), payload("two")]))
    assert [r.json()["answers"]["q"]["state"] for r in responses] == ["one", "two"]
    assert agent.batches == [["one", "two"]]
    assert all("X-Decision-Cache" not in r.headers for r in responses)


def test_verified_hit_finishes_while_model_miss_is_blocked():
    router, agent = configured_router()
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                miss = asyncio.create_task(client.post("/v1/systemone", json=payload("block-model")))
                try:
                    await wait_for(agent.entered)
                    hit = await asyncio.wait_for(client.post("/v1/systemone", json=payload()), 1)
                    assert hit.status_code == 200 and hit.headers["X-Decision-Cache"] == "hit"
                    assert not agent.release.is_set()
                finally:
                    agent.release.set()
                assert (await miss).status_code == 200
    asyncio.run(run())


def test_required_replay_errors_are_terminal_and_sanitized():
    calls = []

    def probe(*args, **kwargs):
        calls.append(1)
        raise DecisionCacheError("private manifest and storage location")

    router, agent = configured_router(probe)
    responses = asyncio.run(drive(serve.create_app(router), [payload(), payload()]))
    assert [r.status_code for r in responses] == [503, 503]
    assert all(r.json() == {"detail": "required decision storage is unavailable"} for r in responses)
    assert all(r.headers["Retry-After"] == "1" for r in responses)
    assert len(calls) == 2 and not agent.calls and not agent.batches


def test_unconfigured_router_retains_original_path():
    router, agent = configured_router(lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected replay")))
    router._replay_path = None
    responses = asyncio.run(drive(serve.create_app(router), [payload("one"), payload("two")]))
    assert all(r.status_code == 200 for r in responses)
    assert agent.batches == [["one", "two"]]
    assert all("cache;dur=" not in r.headers["Server-Timing"] for r in responses)


def test_replay_queue_cancellation_remains_bounded(monkeypatch):
    from laya import _microbatch

    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "2")
    entered, release = threading.Event(), threading.Event()
    queues, calls = [], []
    original = _microbatch.Microbatcher

    class Observed(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queues.append(self)

    def probe(state, questions, **kwargs):
        calls.append(state)
        entered.set()
        assert release.wait(5), "test did not release storage"
        return result(state)

    monkeypatch.setattr(_microbatch, "Microbatcher", Observed)
    router, _ = configured_router(probe)
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                active = asyncio.create_task(client.post("/v1/systemone", json=payload()))
                try:
                    await wait_for(entered)
                    active.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await active
                    for i in range(2):
                        queued = asyncio.create_task(client.post("/v1/systemone", json=payload(str(i))))
                        for _ in range(1000):
                            if queues[0].queue.qsize() == i + 1:
                                break
                            await asyncio.sleep(0.001)
                        assert queues[0].queue.qsize() == i + 1
                        queued.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await queued
                    overflow = await client.post("/v1/systemone", json=payload("overflow"))
                    assert overflow.status_code == 503 and overflow.headers["Retry-After"] == "1"
                    assert queues[0].queue.qsize() == 2 and calls == ["cached"]
                finally:
                    release.set()
    asyncio.run(run())


def test_lifespan_restart_recreates_cache_worker():
    router, agent = configured_router()
    app = serve.create_app(router)
    for _ in range(2):
        response, = asyncio.run(drive(app, [payload()]))
        assert response.status_code == 200 and response.headers["X-Decision-Cache"] == "hit"
    assert not agent.calls


def test_auth_and_invalid_body_are_refused_before_storage(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "secret")
    probes = []
    router, _ = configured_router(lambda *a, **kw: probes.append(1))
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                assert (await client.post("/v1/systemone", json=payload())).status_code == 401
                assert (await client.post("/v1/systemone", json={},
                                          headers={"Authorization": "Bearer secret"})).status_code == 400
    asyncio.run(run())
    assert probes == []


@pytest.mark.parametrize("flag", ["LAYA_CACHE_REQUIRE_EXISTING", "LAYA_CACHE_EXPECTED_STORE_ID"])
def test_store_guards_need_sqlite_path(monkeypatch, flag):
    monkeypatch.setenv(flag, "1" if flag.endswith("EXISTING") else "store-id")
    with pytest.raises(ValueError, match="LAYA_CACHE_PATH"):
        serve._configure_decision_cache(Router())


def test_invalid_store_guard_boolean_fails_startup(monkeypatch):
    monkeypatch.setenv("LAYA_CACHE_REQUIRE_EXISTING", "tru")
    with pytest.raises(ValueError, match="LAYA_CACHE_REQUIRE_EXISTING"):
        serve._configure_decision_cache(Router())


@pytest.mark.parametrize("memory_enabled", [False, True])
def test_replay_manifest_requires_persistent_sqlite_cache(monkeypatch, memory_enabled):
    monkeypatch.setenv("LAYA_REPLAY_MANIFEST", "policy.json")
    if memory_enabled:
        monkeypatch.setenv("LAYA_CACHE", "1")
    with pytest.raises(ValueError, match="LAYA_REPLAY_MANIFEST"):
        serve._configure_decision_cache(Router())


@pytest.mark.parametrize("raw", ["0", "false", "NO", " off "])
def test_http_fast_path_can_be_disabled_independently(monkeypatch, raw):
    monkeypatch.setenv("LAYA_HTTP_REPLAY_FAST_PATH", raw)
    router, agent = configured_router(lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected HTTP probe")))
    response, = asyncio.run(drive(serve.create_app(router), [payload("one")]))
    assert response.status_code == 200 and "X-Decision-Cache" not in response.headers
    assert agent.calls == ["one"]


def test_invalid_http_fast_path_flag_fails_startup(monkeypatch):
    monkeypatch.setenv("LAYA_HTTP_REPLAY_FAST_PATH", "tru")
    with pytest.raises(ValueError, match="LAYA_HTTP_REPLAY_FAST_PATH"):
        serve.create_app(Router())


def test_tournament_mode_keeps_original_path_and_reports_fast_path_disabled(monkeypatch):
    monkeypatch.setenv("LAYA_TOURNAMENT_GROUP", "2")
    router, agent = configured_router(lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected HTTP probe")))
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                response = await client.post("/v1/systemone", json=payload())
                assert response.status_code == 200 and "X-Decision-Cache" not in response.headers
                status = (await client.get("/health")).json()["model_free_replay"]
                assert status == {"enabled": True, "http_fast_path": False}
    asyncio.run(run())
    assert agent.calls == ["cached"]


def approved_store(tmp_path, ttl=None):
    onnx = pytest.importorskip("onnx")
    from laya.checkpoint import _checkpoint_signature

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    graph = checkpoint / "model.onnx"
    onnx.save(onnx.helper.make_model(onnx.helper.make_graph([], "identity-fixture", [], [])), str(graph))
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True, ttl=ttl,
                          maxsize=None, fingerprint="http-policy-v1")
    options = {"models": {"english": str(checkpoint)}, "backend": "onnx",
               "onnx_paths": {"english": str(graph)}}
    router = Router(hooks=[cache], **options)
    _, agent = configured_router()
    agent._checkpoint_signature = _checkpoint_signature(str(checkpoint), str(graph), {})
    router.attach("english", agent)
    initial = router.predict("cached", QUESTIONS, model="english")
    manifest = tmp_path / "policy.json"
    identity = router.export_replay_manifest(str(manifest))
    cache.close()
    return options, manifest, identity, initial


def test_exported_manifest_replays_http_hit_without_loading_checkpoint(tmp_path):
    options, manifest, identity, initial = approved_store(tmp_path)
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True, maxsize=None,
                          fingerprint="http-policy-v1", require_existing=True,
                          expected_store_id=identity["store_id"])
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)

    def never_load(_):
        raise AssertionError("a verified HTTP hit loaded model weights")

    router._build = never_load
    try:
        responses = asyncio.run(drive(serve.create_app(router), [payload(), payload(min_confidence=0.9)]))
        assert all(r.status_code == 200 and r.headers["X-Decision-Cache"] == "hit" for r in responses)
        assert responses[0].json()["answers"] == initial["answers"]
        assert responses[1].json()["answers"]["q"]["abstention"] == "abstained"
        assert router.loaded == [] and cache.cache_info()["hits"] == 2
    finally:
        cache.close()


def test_deployment_replay_and_store_guard_flags_are_forwarded(tmp_path, monkeypatch):
    import json

    options, manifest, identity, _ = approved_store(tmp_path)
    monkeypatch.setenv("LAYA_BACKEND", "onnx")
    monkeypatch.setenv("LAYA_PRELOAD", "0")
    monkeypatch.setenv("LAYA_EXTRA_MODELS", json.dumps(options["models"]))
    monkeypatch.setenv("LAYA_ONNX_PATHS", json.dumps(options["onnx_paths"]))
    monkeypatch.setenv("LAYA_REPLAY_MANIFEST", " " + str(manifest) + " ")
    monkeypatch.setenv("LAYA_CACHE_PATH", str(tmp_path / "decisions.sqlite"))
    monkeypatch.setenv("LAYA_CACHE_REQUIRED", "1")
    monkeypatch.setenv("LAYA_CACHE_REQUIRE_EXISTING", "true")
    monkeypatch.setenv("LAYA_CACHE_EXPECTED_STORE_ID", identity["store_id"])
    monkeypatch.setenv("LAYA_CACHE_FINGERPRINT", "http-policy-v1")
    router = serve.build_router()
    try:
        assert router.replay_manifest == str(manifest.resolve())
        assert router._laya_decision_cache.require_existing is True
        response, = asyncio.run(drive(serve.create_app(router), [payload()]))
        assert response.status_code == 200 and response.headers["X-Decision-Cache"] == "hit"
        assert router.loaded == []
        report = serve._decision_cache_report(router)["decision_cache"]
        assert report["store_id"] == identity["store_id"] and report["require_existing"] is True
    finally:
        serve._close_decision_cache(router)


def test_builder_rejects_manifest_store_mismatch_and_closes_installed_cache(tmp_path, monkeypatch):
    import sqlite3

    options, manifest, _, _ = approved_store(tmp_path)
    monkeypatch.setenv("LAYA_CACHE_PATH", str(tmp_path / "different.sqlite"))
    monkeypatch.setenv("LAYA_CACHE_FINGERPRINT", "http-policy-v1")
    router = Router(replay_manifest=str(manifest), **options)
    added, original = [], router.add_hook

    def capture(cache):
        added.append(cache)
        return original(cache)

    monkeypatch.setattr(router, "add_hook", capture)
    with pytest.raises(DecisionCacheError, match="store|configuration"):
        serve._configure_decision_cache(router)
    assert router.hooks == [] and len(added) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        added[0].cache_info()


def test_custom_hook_chain_has_no_dedicated_cache_worker(tmp_path):
    from laya.checkpoint import _checkpoint_signature

    options, manifest, _, _ = approved_store(tmp_path)
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True,
                          maxsize=None, fingerprint="http-policy-v1")

    class Observer:
        starts = 0

        def on_predict_start(self, context):
            self.starts += 1

    observer = Observer()
    router = Router(hooks=[cache, observer], replay_manifest=str(manifest), **options)
    _, agent = configured_router()
    agent._checkpoint_signature = _checkpoint_signature(options["models"]["english"],
                                                         options["onnx_paths"]["english"], {})
    router.attach("english", agent)
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                response = await client.post("/v1/systemone", json=payload())
                assert response.status_code == 200 and "X-Decision-Cache" not in response.headers
                assert "cache;dur=" not in response.headers["Server-Timing"]
                assert (await client.get("/health")).json()["model_free_replay"]["http_fast_path"] is False
    try:
        asyncio.run(run())
        assert observer.starts == 1 and cache.cache_info()["hits"] == 1 and not agent.calls
    finally:
        cache.close()


def test_real_replay_storage_failure_returns_sanitized_503(tmp_path, monkeypatch):
    options, manifest, _, _ = approved_store(tmp_path)
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True,
                          maxsize=None, fingerprint="http-policy-v1")
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)
    attempts = []
    monkeypatch.setattr(router, "_build", lambda _: attempts.append("model"))

    def unavailable(*args):
        raise OSError("private database path")

    monkeypatch.setattr(cache._store, "get", unavailable)
    try:
        response, = asyncio.run(drive(serve.create_app(router), [payload()]))
        assert response.status_code == 503
        assert response.json() == {"detail": "required decision storage is unavailable"}
        assert attempts == []
    finally:
        cache.close()


def test_verified_replay_keeps_invalid_routing_controls_as_client_errors(tmp_path):
    options, manifest, _, _ = approved_store(tmp_path)
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True,
                          maxsize=None, fingerprint="http-policy-v1")
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)
    request = payload(task="unknown-task")
    request.pop("model")
    try:
        response, = asyncio.run(drive(serve.create_app(router), [request]))
        assert response.status_code == 422
        assert "unknown-task" in response.text and router.loaded == []
    finally:
        cache.close()


def test_verified_http_hit_survives_idle_model_unload(tmp_path, monkeypatch):
    from laya.checkpoint import _checkpoint_signature

    options, manifest, _, initial = approved_store(tmp_path)
    monkeypatch.setenv("LAYA_IDLE_UNLOAD_SECONDS", "0.05")
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True,
                          maxsize=None, fingerprint="http-policy-v1")
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)
    _, agent = configured_router()
    agent._checkpoint_signature = _checkpoint_signature(options["models"]["english"],
                                                         options["onnx_paths"]["english"], {})
    router.attach("english", agent)
    app = serve.create_app(router)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                for _ in range(500):
                    if not router.loaded:
                        break
                    await asyncio.sleep(0.01)
                assert router.loaded == [], "idle reaper did not unload the model"
                response = await client.post("/v1/systemone", json=payload())
                assert response.status_code == 200 and response.headers["X-Decision-Cache"] == "hit"
                assert response.json()["answers"] == initial["answers"]
                assert router.loaded == [] and agent.calls == []
    try:
        asyncio.run(run())
    finally:
        cache.close()


@pytest.mark.parametrize("renew_failure", [False, True])
def test_http_verified_replay_preserves_retention_renewal(tmp_path, monkeypatch, renew_failure):
    import laya.consistency as consistency

    clock = [100.0]
    monkeypatch.setattr(consistency, "_now", lambda: clock[0])
    options, manifest, _, _ = approved_store(tmp_path, ttl=160)
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True, ttl=160, renew_on_hit=True,
                          maxsize=None, fingerprint="http-policy-v1")
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)
    renew, calls = cache._store.renew, []

    def recording_renew(*args):
        calls.append(1)
        if renew_failure:
            raise OSError("private storage path")
        return renew(*args)

    monkeypatch.setattr(cache._store, "renew", recording_renew)
    try:
        clock[0] = 120.0
        response, = asyncio.run(drive(serve.create_app(router), [payload()]))
        assert response.status_code == (503 if renew_failure else 200)
        assert calls == [1] and router.loaded == []
        if not renew_failure:
            clock[0] = 265.0  # beyond the original expiry of 260; the verified hit renewed it
            later, = asyncio.run(drive(serve.create_app(router), [payload()]))
            assert later.status_code == 200 and later.headers["X-Decision-Cache"] == "hit"
    finally:
        cache.close()


def test_cache_batch_errors_are_isolated_per_request():
    def probe(state, questions, **kwargs):
        if state == "invalid":
            raise ValueError("only this request is invalid")
        return result(state)

    router, agent = configured_router(probe)
    responses = asyncio.run(drive(serve.create_app(router), [payload("one"), payload("invalid"), payload("two")]))
    assert [r.status_code for r in responses] == [200, 422, 200]
    assert all("only this" not in r.text for r in (responses[0], responses[2]))
    assert not agent.calls


def test_shutdown_settles_active_and_queued_cache_requests():
    entered, release = threading.Event(), threading.Event()

    def probe(state, questions, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release storage"
        return result(state)

    router, _ = configured_router(probe)
    app = serve.create_app(router)

    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            active = asyncio.create_task(client.post("/v1/systemone", json=payload()))
            await wait_for(entered)
            queued = asyncio.create_task(client.post("/v1/systemone", json=payload("queued")))
            await asyncio.sleep(0)
            # Closing the executor drains an uncancellable SQLite operation. Its completion
            # comes from another thread; the event loop need not drive the test release.
            timer = threading.Timer(0.05, release.set)
            timer.start()
            try:
                await lifespan.__aexit__(None, None, None)
                responses = await asyncio.gather(active, queued)
                assert [r.status_code for r in responses] == [503, 503]
            finally:
                release.set()
                timer.join()
    asyncio.run(run())


@pytest.mark.parametrize("microbatch", [False, True])
def test_replay_miss_resuming_during_shutdown_cannot_start_model_work(tmp_path, monkeypatch, microbatch):
    from laya import _microbatch
    from laya.checkpoint import _checkpoint_signature

    monkeypatch.setenv("LAYA_MICROBATCH_MAX_STATES", "8" if microbatch else "0")
    options, manifest, _, _ = approved_store(tmp_path)
    cache = DecisionCache(str(tmp_path / "decisions.sqlite"), required=True,
                          maxsize=None, fingerprint="http-policy-v1")
    router = Router(hooks=[cache], replay_manifest=str(manifest), **options)
    _, agent = configured_router()
    agent._checkpoint_signature = _checkpoint_signature(options["models"]["english"],
                                                         options["onnx_paths"]["english"], {})
    router.attach("english", agent)
    original, queues, state = _microbatch.Microbatcher, [], {}

    class StopBeforeMissResumes(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queues.append(self)

        @staticmethod
        def _finish(job, value=None, error=None):
            if job.key == "replay" and value is not None and value[0] is None and "shutdown" not in state:
                # Resolve the miss only after shutdown is ready to run. No await is added
                # to application code: the HTTP continuation is an ordinary future wakeup.
                state["shutdown"] = asyncio.create_task(state["lifespan"].__aexit__(None, None, None))
            original._finish(job, value, error)

    monkeypatch.setattr(_microbatch, "Microbatcher", StopBeforeMissResumes)
    app = serve.create_app(router)

    async def run():
        state["lifespan"] = app.router.lifespan_context(app)
        await state["lifespan"].__aenter__()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                response = await client.post("/v1/systemone", json=payload("new-miss"))
                await state["shutdown"]
                assert response.status_code == 503 and response.headers["Retry-After"] == "1"
                assert all(queue.closed and queue.task.done() for queue in queues)
                assert len(queues) == 1 and not agent.calls and not agent.batches
        finally:
            if "shutdown" not in state:
                await state["lifespan"].__aexit__(None, None, None)
            for queue in queues:
                await queue.close()
    try:
        asyncio.run(run())
    finally:
        cache.close()


def test_deferred_runtime_captures_device_and_thread_values(monkeypatch):
    from laya.mcp.device import env_device

    monkeypatch.setenv("LAYA_REPLAY_MANIFEST", "approved.json")
    monkeypatch.setenv("LAYA_THREADS", "3")
    monkeypatch.setenv("LAYA_DEVICE", " CPU ")
    calls = []
    fake_torch = SimpleNamespace(device=lambda value: calls.append(("device", value)) or value,
                                 set_num_threads=lambda value: calls.append(("threads", value)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    device, prepare = serve._router_runtime_setup("torch", None)
    assert device == "cpu" and calls == [] and env_device(validate=False) == "cpu"
    monkeypatch.setenv("LAYA_THREADS", "9")
    monkeypatch.setenv("LAYA_DEVICE", "cuda:258")
    prepare()
    assert calls == [("device", "cpu"), ("threads", 3)]


def test_deferred_runtime_still_refuses_torch_device_rewrites(monkeypatch):
    monkeypatch.setenv("LAYA_REPLAY_MANIFEST", "approved.json")
    monkeypatch.setenv("LAYA_DEVICE", "cuda:258")
    monkeypatch.setenv("LAYA_THREADS", "3")
    device, prepare = serve._router_runtime_setup("torch", None)
    assert device == "cuda:258"
    fake_torch = SimpleNamespace(device=lambda value: "cuda:2",
                                 set_num_threads=lambda value: pytest.fail("invalid device applied threads"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(ValueError, match="LAYA_DEVICE"):
        prepare()


@pytest.mark.parametrize("raw", ["0", "-1", "2.5", "many"])
def test_deferred_runtime_refuses_invalid_threads_before_model_load(monkeypatch, raw):
    monkeypatch.setenv("LAYA_REPLAY_MANIFEST", "approved.json")
    monkeypatch.setenv("LAYA_THREADS", raw)
    with pytest.raises(ValueError, match="LAYA_THREADS"):
        serve._router_runtime_setup("torch", None)


@pytest.mark.parametrize("mode", ["default", "2"])
def test_http_builder_and_health_do_not_import_ml_libraries_for_verified_hits(tmp_path, mode):
    from laya import Agent
    from laya.checkpoint import _checkpoint_signature

    directory = tmp_path / "torch-checkpoint"
    directory.mkdir()
    (directory / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    weights = directory / "model.safetensors"
    weights.write_bytes(b"controlled identity only; never loaded")
    cache_path, manifest = tmp_path / "cold.sqlite", tmp_path / "cold.json"
    cache = DecisionCache(str(cache_path), required=True, maxsize=None, fingerprint="cold-http-v1")
    router = Router(models={"english": str(directory)}, hooks=[cache])
    agent = object.__new__(Agent)
    agent.cfg, agent.model_id, agent.revision = Predictor.cfg, Predictor.model_id, Predictor.revision
    agent.calls, agent.batches = [], []
    agent.system_one = MethodType(Predictor.system_one, agent)
    agent.predict_batch = MethodType(Predictor.predict_batch, agent)
    agent._checkpoint_signature = _checkpoint_signature(str(directory), str(weights), {})
    router.attach("english", agent)
    try:
        router.predict("cached", QUESTIONS, model="english")
        approval = router.export_replay_manifest(str(manifest))
    finally:
        cache.close()
    program = """
import asyncio, importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class DenyML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','transformers','onnx','onnxruntime','tokenizers','safetensors'):
            raise AssertionError('ML dependency import during verified HTTP replay: ' + fullname)
sys.meta_path.insert(0, DenyML())
from laya import serve
import httpx
router = serve.build_router()
app = serve.create_app(router)
async def run():
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://local') as client:
            response = await client.post('/v1/systemone',json=json.loads(sys.argv[2]))
            assert response.status_code == 200, response.text
            assert response.headers['X-Decision-Cache'] == 'hit'
            health = (await client.get('/health')).json()
            assert health['loaded'] == [] and health['device'] == 'cpu' and health['device_is_preference']
            assert health['model_free_replay']['http_fast_path'] is True
try:
    asyncio.run(run())
finally:
    serve._close_decision_cache(router)
print('verified cold HTTP replay without ML imports')
"""
    environment = os.environ.copy()
    environment.update(LAYA_BACKEND="torch", LAYA_PRELOAD="0", LAYA_DEVICE=" CPU ", LAYA_THREADS="2",
                       LAYA_PERFORMANCE_MODE=mode, LAYA_CACHE_PATH=str(cache_path), LAYA_CACHE_REQUIRED="1",
                       LAYA_CACHE_FINGERPRINT="cold-http-v1", LAYA_CACHE_REQUIRE_EXISTING="1",
                       LAYA_CACHE_EXPECTED_STORE_ID=approval["store_id"], LAYA_REPLAY_MANIFEST=str(manifest),
                       LAYA_EXTRA_MODELS=json.dumps({"english": str(directory)}), HF_HUB_OFFLINE="1")
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run([sys.executable, "-c", program, str(root), json.dumps(payload())],
                             env=environment, cwd=root, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stdout + process.stderr
