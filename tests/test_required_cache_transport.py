"""Required local cache failures must surface as errors, never alternate predictions."""
import pytest

from veya import DecisionCache, Router
from veya import serve
from veya.consistency import _MemoryStore
from veya.mcp.tools import ToolError, veya_predict, veya_predict_batch


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
    for name in ("VEYA_CACHE", "VEYA_CACHE_PATH", "VEYA_CACHE_REQUIRED", "VEYA_CACHE_FINGERPRINT",
                 "VEYA_API_KEY", "VEYA_MICROBATCH_MAX_STATES", "VEYA_TOURNAMENT_GROUP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VEYA_PRELOAD", "0")


@pytest.mark.parametrize("raw", ["1", "true", "YES", " on "])
def test_required_deployment_is_explicit_and_reported(monkeypatch, raw):
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv("VEYA_CACHE_REQUIRED", raw)
    router = Router()
    serve._configure_decision_cache(router)
    try:
        assert router._veya_decision_cache.required
        assert serve._decision_cache_report(router)["decision_cache"]["required"] is True
    finally:
        serve._close_decision_cache(router)


@pytest.mark.parametrize("raw", ["0", "false", "NO", " off ", ""])
def test_optional_deployment_remains_default(monkeypatch, raw):
    monkeypatch.setenv("VEYA_CACHE", "1")
    monkeypatch.setenv("VEYA_CACHE_REQUIRED", raw)
    router = Router()
    serve._configure_decision_cache(router)
    try:
        assert not router._veya_decision_cache.required
    finally:
        serve._close_decision_cache(router)


def test_required_deployment_cannot_silently_disable_caching(monkeypatch):
    monkeypatch.setenv("VEYA_CACHE_REQUIRED", "true")
    with pytest.raises(ValueError, match="VEYA_CACHE_REQUIRED"):
        serve._configure_decision_cache(Router())
    monkeypatch.setenv("VEYA_CACHE_REQUIRED", "tru")
    with pytest.raises(ValueError, match="VEYA_CACHE_REQUIRED"):
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
            veya_predict_batch([request], router=router)
        else:
            veya_predict(**request, router=router)
    assert failure.value.code == "decision_cache_unavailable"
    assert "private-storage-location" not in str(failure.value)
