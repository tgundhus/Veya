"""Local HTTP/MCP cache configuration and retention without checkpoint downloads."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from veya import Router
from veya import serve


QUESTIONS = {"q": {"type": "noul", "instructions": "Is this a refund?"}}
REQUEST = {"state": "refund please", "questions": QUESTIONS, "model": "english"}


class CountingAgent:
    cfg = {"max_len": 512, "head_max_len": 256}
    model_id_or_path = "test-checkpoint"
    revision = "test-revision"

    def __init__(self, answer=0.9):
        self.calls = 0
        self.answer = answer

    def system_one(self, state, questions, **kwargs):
        self.calls += 1
        return {"model": "test", "answers": {"q": {"type": "noul", "noul": self.answer}},
                "usage": {"input_tokens": 8, "output_tokens": 0}}

    def predict_batch(self, states, questions, **kwargs):
        return [self.system_one(state, questions, **kwargs) for state in states]


@pytest.fixture(autouse=True)
def local_environment(monkeypatch):
    for name in ("VEYA_CACHE", "VEYA_CACHE_PATH", "VEYA_CACHE_TTL", "VEYA_CACHE_MAXSIZE",
                 "VEYA_CACHE_FINGERPRINT", "VEYA_CACHE_RENEW_ON_HIT", "VEYA_API_KEY",
                 "VEYA_EXTRA_MODELS", "VEYA_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VEYA_PRELOAD", "0")


def attached_router(answer=0.9):
    agent = CountingAgent(answer)
    router = Router()
    router.attach("english", agent)
    return router, agent


def test_cache_is_opt_in_and_retention_settings_are_validated(monkeypatch):
    router, _ = attached_router()
    serve._configure_decision_cache(router)
    assert serve._decision_cache_report(router) == {}
    monkeypatch.setenv("VEYA_CACHE", "1")
    serve._configure_decision_cache(router)
    try:
        report = serve._decision_cache_report(router)["decision_cache"]
        assert report["ttl_seconds"] == 86400
        assert report["maxsize"] == 10000
        assert report["renew_on_hit"] is False
        assert router._veya_decision_cache.fingerprint is None
    finally:
        serve._close_decision_cache(router)


@pytest.mark.parametrize("name,value", [
    ("VEYA_CACHE_TTL", "0"), ("VEYA_CACHE_TTL", "nan"), ("VEYA_CACHE_TTL", "inf"),
    ("VEYA_CACHE_TTL", "-1"), ("VEYA_CACHE_MAXSIZE", "0"), ("VEYA_CACHE_MAXSIZE", "2.5"),
    ("VEYA_CACHE_MAXSIZE", ""), ("VEYA_CACHE_MAXSIZE", "null"), ("VEYA_CACHE_MAXSIZE", "-1"),
])
def test_invalid_retention_refuses_configuration(monkeypatch, name, value):
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        serve.build_router()


@pytest.mark.parametrize("maxsize,expected", [("32", 32), ("none", None), (" NONE ", None)])
def test_capacity_and_policy_identity_are_forwarded(monkeypatch, maxsize, expected):
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv("VEYA_CACHE_MAXSIZE", maxsize)
    monkeypatch.setenv("VEYA_CACHE_FINGERPRINT", " policy-v1 ")
    router, _ = attached_router()
    serve._configure_decision_cache(router)
    try:
        assert router._veya_decision_cache.maxsize == expected
        assert router._veya_decision_cache.fingerprint == "policy-v1"
    finally:
        serve._close_decision_cache(router)


def test_empty_policy_identity_keeps_automatic_fingerprinting(monkeypatch):
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv("VEYA_CACHE_FINGERPRINT", " \t ")
    router, _ = attached_router()
    serve._configure_decision_cache(router)
    try:
        assert router._veya_decision_cache.fingerprint is None
    finally:
        serve._close_decision_cache(router)


def test_durable_policy_replays_borderline_decision_after_restart(monkeypatch, tmp_path):
    import veya.consistency as consistency

    clock = [100.0]
    monkeypatch.setattr(consistency, "_now", lambda: clock[0])
    monkeypatch.setenv("VEYA_CACHE_PATH", str(tmp_path / "durable.sqlite"))
    monkeypatch.setenv("VEYA_CACHE_TTL", "none")
    monkeypatch.setenv("VEYA_CACHE_MAXSIZE", "none")
    monkeypatch.setenv("VEYA_CACHE_FINGERPRINT", "policy-v1")
    first_router, first_agent = attached_router(0.51)
    serve._configure_decision_cache(first_router)
    try:
        with TestClient(serve.create_app(first_router)) as client:
            first = client.post("/v1/systemone", json=REQUEST)
            assert first.status_code == 200
            assert first.json()["answers"]["q"]["noul"] == 0.51
            assert first_agent.calls == 1
    finally:
        serve._close_decision_cache(first_router)

    clock[0] += 10 * 365 * 86400
    second_router, second_agent = attached_router(0.49)
    second_agent.revision = "replacement-revision"
    serve._configure_decision_cache(second_router)
    try:
        with TestClient(serve.create_app(second_router)) as client:
            repeat = client.post("/v1/systemone", json=REQUEST)
            assert repeat.status_code == 200
            assert repeat.json()["answers"]["q"]["noul"] == 0.51
            assert second_agent.calls == 0
            report = client.get("/health").json()["decision_cache"]
            assert report["ttl_seconds"] is None and report["maxsize"] is None
    finally:
        serve._close_decision_cache(second_router)

    # A deliberate policy-version change permits a fresh decision.
    monkeypatch.setenv("VEYA_CACHE_FINGERPRINT", "policy-v2")
    serve._configure_decision_cache(second_router)
    try:
        assert second_router.predict(**REQUEST)["answers"]["q"]["noul"] == 0.49
        assert second_agent.calls == 1
    finally:
        serve._close_decision_cache(second_router)


def test_http_replays_then_expires_and_hides_cache_details_from_anonymous_probe(monkeypatch):
    import veya.consistency as consistency

    clock = [100.0]
    monkeypatch.setattr(consistency, "_now", lambda: clock[0])
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv("VEYA_CACHE_TTL", "2")
    monkeypatch.setenv("VEYA_API_KEY", "test-key")
    router, agent = attached_router()
    serve._configure_decision_cache(router)
    try:
        with TestClient(serve.create_app(router)) as client:
            auth = {"Authorization": "Bearer test-key"}
            first = client.post("/v1/systemone", json=REQUEST, headers=auth)
            repeat = client.post("/v1/systemone", json=REQUEST, headers=auth)
            assert first.status_code == repeat.status_code == 200
            assert first.json() == repeat.json()
            assert agent.calls == 1
            assert client.get("/health").json() == {"status": "ok"}
            report = client.get("/health", headers=auth).json()["decision_cache"]
            assert report["hits"] == report["misses"] == 1
            clock[0] += 3
            assert client.post("/v1/systemone", json=REQUEST, headers=auth).status_code == 200
            assert agent.calls == 2
    finally:
        serve._close_decision_cache(router)


def test_sqlite_cache_survives_http_lifespan_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("VEYA_CACHE_PATH", str(tmp_path / "decisions.sqlite"))
    router, agent = attached_router()
    serve._configure_decision_cache(router)
    monkeypatch.setattr(serve, "build_router", lambda: router)
    app = serve.create_app()
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=REQUEST).status_code == 200
    assert router._veya_decision_cache is None
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=REQUEST).status_code == 200
        assert agent.calls == 1
        assert client.get("/health").json()["decision_cache"]["hits"] == 1
    assert router._veya_decision_cache is None


@pytest.mark.parametrize("maxsize,expected", [("32", 32), ("none", None)])
def test_mcp_uses_the_same_retention_and_loaded_limit(monkeypatch, maxsize, expected):
    pytest.importorskip("mcp")
    from veya.mcp import server
    from veya.mcp.tools import veya_status

    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv("VEYA_CACHE_TTL", "none")
    monkeypatch.setenv("VEYA_CACHE_MAXSIZE", maxsize)
    monkeypatch.setenv("VEYA_CACHE_FINGERPRINT", "policy-v1")
    monkeypatch.setenv("VEYA_CACHE_RENEW_ON_HIT", "1")
    monkeypatch.setenv("VEYA_MAX_LOADED", "3")
    router = server._ensure_router()
    try:
        assert router.max_loaded == 3
        report = veya_status(router=router)["decision_cache"]
        assert report["ttl_seconds"] is None
        assert report["maxsize"] == expected
        assert router._veya_decision_cache.fingerprint == "policy-v1"
        assert report["renew_on_hit"] is True
    finally:
        serve._close_decision_cache(router)


def test_http_lifespan_recovers_cache_after_failed_restart(monkeypatch):
    monkeypatch.setenv("VEYA_CACHE", "1")
    router, agent = attached_router()
    serve._configure_decision_cache(router)
    monkeypatch.setattr(serve, "build_router", lambda: router)
    app = serve.create_app()
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=REQUEST).status_code == 200
    assert router._veya_decision_cache is None

    monkeypatch.setenv("VEYA_CACHE_TTL", "0")
    with pytest.raises(ValueError, match="VEYA_CACHE_TTL"):
        with TestClient(app):
            pass
    assert router._veya_decision_cache is None

    monkeypatch.setenv("VEYA_CACHE_TTL", "86400")
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=REQUEST).status_code == 200
        assert client.post("/v1/systemone", json=REQUEST).status_code == 200
        report = client.get("/health").json()["decision_cache"]
        assert report["ttl_seconds"] == 86400
        assert report["hits"] == 1
        assert agent.calls == 2
    assert router._veya_decision_cache is None


def test_mcp_tools_accept_locally_registered_checkpoints(monkeypatch):
    pytest.importorskip("mcp")
    from veya.mcp import server
    from veya.mcp.tools import veya_predict, veya_predict_batch, veya_route, veya_route_batch

    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setenv("VEYA_EXTRA_MODELS", '{"refund-v2": "local-checkpoint"}')
    router = server._ensure_router()
    router.attach("refund-v2", CountingAgent())
    item = {"state": {"text": "refund please"}, "questions": QUESTIONS, "model": "refund-v2"}
    assert veya_predict(**item, router=router)["routing"]["model"] == "refund-v2"
    assert veya_route(**item, router=router)["model"] == "refund-v2"
    assert veya_predict_batch([item], router=router)["requests"][0]["routing"]["model"] == "refund-v2"
    assert veya_route_batch([item], router=router)["decisions"][0]["model"] == "refund-v2"


def test_registered_default_honors_the_same_casing_as_router(monkeypatch):
    monkeypatch.setenv("VEYA_EXTRA_MODELS", '{"Refund-V2": "local-checkpoint"}')
    monkeypatch.setenv("VEYA_DEFAULT_MODEL", " REFUND-V2 ")
    assert serve.build_router().default == "refund-v2"
