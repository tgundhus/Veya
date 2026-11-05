"""Required local cache failures must surface as errors, never alternate predictions."""
import pytest

from laya import DecisionCache, Router
from laya import serve
from laya.consistency import _MemoryStore
from laya.mcp.tools import ToolError, laya_predict, laya_predict_batch


QUESTIONS = {"keep": {"type": "noul", "instructions": "Keep this?"}}
REQUEST = {"state": "same input", "questions": QUESTIONS, "model": "english"}


class Predictor:
    cfg = {"max_len": 512, "head_max_len": 192}
    model_id = "local-test-model"

    def system_one(self, state, questions, **kwargs):
        return {"answers": {name: {"type": "noul", "noul": 0.49} for name in questions},
                "usage": {"input_tokens": 1, "output_tokens": 0}}

    def predict_batch(self, states, questions, **kwargs):
        return [self.system_one(state, questions, **kwargs) for state in states]


class BrokenStore(_MemoryStore):
    def __init__(self, operation):
        super().__init__(None)
        self.operation = operation

    def get(self, *args):
        if self.operation == "read":
            raise OSError("private-storage-location")
        if self.operation == "decode":
            return b"invalid-stored-decision", 0.0
        return super().get(*args)

    def add(self, *args):
        raise OSError("private-storage-location")


def broken_router(operation):
    cache = DecisionCache(store=BrokenStore(operation), required=True)
    router = Router(hooks=[cache], hooks_raise=False)
    router.attach("english", Predictor())
    return router


@pytest.fixture(autouse=True)
def local_environment(monkeypatch):
    for name in ("LAYA_CACHE", "LAYA_CACHE_PATH", "LAYA_CACHE_REQUIRED", "LAYA_CACHE_FINGERPRINT",
                 "LAYA_API_KEY", "LAYA_MICROBATCH_MAX_STATES", "LAYA_TOURNAMENT_GROUP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYA_PRELOAD", "0")


@pytest.mark.parametrize("raw", ["1", "true", "YES", " on "])
def test_required_deployment_is_explicit_and_reported(monkeypatch, raw):
    monkeypatch.setenv("LAYA_CACHE", "1")
    monkeypatch.setenv("LAYA_CACHE_REQUIRED", raw)
    router = Router()
    serve._configure_decision_cache(router)
    try:
        assert router._laya_decision_cache.required
        assert serve._decision_cache_report(router)["decision_cache"]["required"] is True
    finally:
        serve._close_decision_cache(router)


@pytest.mark.parametrize("raw", ["0", "false", "NO", " off ", ""])
def test_optional_deployment_remains_default(monkeypatch, raw):
    monkeypatch.setenv("LAYA_CACHE", "1")
    monkeypatch.setenv("LAYA_CACHE_REQUIRED", raw)
    router = Router()
    serve._configure_decision_cache(router)
    try:
        assert not router._laya_decision_cache.required
    finally:
        serve._close_decision_cache(router)


def test_required_deployment_cannot_silently_disable_caching(monkeypatch):
    monkeypatch.setenv("LAYA_CACHE_REQUIRED", "true")
    with pytest.raises(ValueError, match="LAYA_CACHE_REQUIRED"):
        serve._configure_decision_cache(Router())
    monkeypatch.setenv("LAYA_CACHE_REQUIRED", "tru")
    with pytest.raises(ValueError, match="LAYA_CACHE_REQUIRED"):
        serve._configure_decision_cache(Router())


@pytest.mark.parametrize("operation", ["read", "write", "decode"])
@pytest.mark.parametrize("batch", [False, True])
def test_http_required_errors_are_sanitized_and_never_return_answers(operation, batch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    with TestClient(serve.create_app(broken_router(operation))) as client:
        response = client.post("/v1/systemone/batch" if batch else "/v1/systemone",
                               json={"states": ["same input"], "questions": QUESTIONS, "model": "english"}
                               if batch else REQUEST)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json() == {"detail": "required decision storage is unavailable"}
    assert "private-storage-location" not in response.text


@pytest.mark.parametrize("operation", ["read", "write", "decode"])
@pytest.mark.parametrize("batch", [False, True])
def test_mcp_required_errors_are_sanitized(operation, batch):
    router = broken_router(operation)
    request = {**REQUEST, "state": {"text": "same input"}}
    with pytest.raises(ToolError) as failure:
        if batch:
            laya_predict_batch([request], router=router)
        else:
            laya_predict(**request, router=router)
    assert failure.value.code == "decision_cache_unavailable"
    assert "private-storage-location" not in str(failure.value)
