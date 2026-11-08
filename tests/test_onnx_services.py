"""Local ONNX service configuration and lifecycle, without weights or hosted services."""
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veya import Router  # noqa: E402
from veya import serve  # noqa: E402

QUESTIONS = {"q": {"type": "choice", "instructions": "Choose the category",
                   "criteria": {"billing": "billing", "support": "support"}}}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("VEYA_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("VEYA_PRELOAD", "0")


@pytest.fixture
def checkpoint(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    path = tmp_path / "checkpoint"
    tok_dir = path / "tokenizer"
    tok_dir.mkdir(parents=True)
    special = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"]
    tokenizer = Tokenizer(models.WordPiece(dict(zip(special + ["billing", "support", "refund"], range(8))),
                                          unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tok_dir / "tokenizer.json"))
    (tok_dir / "tokenizer_config.json").write_text(json.dumps(dict(zip(
        ["unk_token", "cls_token", "sep_token", "mask_token", "pad_token"], special))), encoding="utf-8")
    (path / "rl_agent_config.json").write_text(json.dumps({"max_len": 96, "head_max_len": 40}), encoding="utf-8")
    (path / "veya.onnx").write_bytes(b"fake local graph")
    return path


def fake_ort():
    class Session:
        def __init__(self, path, *, sess_options, providers):
            assert providers == ["CPUExecutionProvider"]
            self.options = sess_options
            self.calls = 0

        def run(self, names, inputs):
            self.calls += 1
            mask = inputs["marker_mask"]
            return [np.where(mask, np.arange(mask.shape[1], dtype=np.float32), -1e9),
                    np.zeros((len(mask), 2), dtype=np.float32)]

    return types.SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                                 SessionOptions=types.SimpleNamespace, InferenceSession=Session,
                                 GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=99))


def configure(monkeypatch, checkpoint):
    monkeypatch.setenv("VEYA_BACKEND", "onnx")
    monkeypatch.setenv("VEYA_EXTRA_MODELS", json.dumps({"en": str(checkpoint)}))
    monkeypatch.setenv("VEYA_ONNX_PATHS", json.dumps({"en": str(checkpoint / "veya.onnx")}))
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort())


def test_operator_configuration_freezes_paths_and_backend(checkpoint, monkeypatch):
    configure(monkeypatch, checkpoint)
    monkeypatch.setenv("VEYA_THREADS", "3")
    router = serve.build_router()
    monkeypatch.setenv("VEYA_BACKEND", "invalid")
    monkeypatch.setenv("VEYA_ONNX_PATHS", "invalid JSON")
    try:
        assert router.backend == "onnx"
        assert router.onnx_paths == {"english": str(checkpoint / "veya.onnx")}
        assert router.load("en").session.options.intra_op_num_threads == 3
    finally:
        router.unload()


@pytest.mark.parametrize("raw", ["", "{}", "[]", "null", "broken"])
def test_onnx_requires_an_operator_graph_map(raw, monkeypatch):
    monkeypatch.setenv("VEYA_BACKEND", "onnx")
    monkeypatch.setenv("VEYA_ONNX_PATHS", raw)
    with pytest.raises(ValueError, match="VEYA_ONNX_PATHS"):
        serve.build_router()


def test_alias_collisions_and_unpaired_graphs_are_configuration_errors(checkpoint):
    graph = str(checkpoint / "veya.onnx")
    with pytest.raises(ValueError, match="more than once"):
        Router(backend="onnx", models={"en": str(checkpoint)}, onnx_paths={"en": graph, "english": graph})
    with pytest.raises(ValueError, match="local checkpoint directory"):
        Router(backend="onnx", onnx_paths={"english": graph})
    with pytest.raises(FileNotFoundError, match="ONNX graph"):
        Router(backend="onnx", models={"en": str(checkpoint)}, onnx_paths={"en": graph + ".missing"})


def test_only_configured_exports_are_preloaded_and_missing_route_is_actionable(checkpoint, monkeypatch):
    configure(monkeypatch, checkpoint)
    monkeypatch.setenv("VEYA_PRELOAD", "1")
    router = serve.build_router()
    assert router.loaded == ["english"]
    with pytest.raises(ValueError, match="no ONNX export configured.*multilingual"):
        router.predict("hello", QUESTIONS, model="multilingual")
    router.unload()


def test_onnx_device_is_validated_without_torch(checkpoint, monkeypatch):
    configure(monkeypatch, checkpoint)
    monkeypatch.setenv("VEYA_DEVICE", "cuda")
    with pytest.raises(ValueError, match="VEYA_DEVICE.*cpu"):
        serve.build_router()


def test_cli_backend_precedes_environment(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    calls = []
    monkeypatch.setenv("VEYA_BACKEND", "onnx")
    monkeypatch.setattr(serve, "create_app", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    serve.main(["--backend", "torch"])
    assert calls == [{"backend": "torch"}]
    assert os.environ["VEYA_BACKEND"] == "onnx"


def test_injected_app_backend_must_match(checkpoint, monkeypatch):
    pytest.importorskip("fastapi")
    configure(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="backend must match"):
        serve.create_app(router=serve.build_router(), backend="torch")


def exercise_no_heavy_imports(path):
    """Executed in a clean subprocess: even a swallowed backend probe fails the check."""
    class ForbiddenImport(BaseException):
        pass

    class NoHeavyImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("torch", "transformers"):
                raise ForbiddenImport("forbidden import: " + fullname)

    sys.meta_path.insert(0, NoHeavyImports())
    assert not any(name.split(".")[0] in ("torch", "transformers") for name in sys.modules)
    for name in tuple(os.environ):
        if name.startswith("VEYA_"):
            os.environ.pop(name)
    os.environ.update(VEYA_BACKEND="onnx", VEYA_PRELOAD="0", VEYA_THREADS="2", VEYA_CACHE="1",
                      VEYA_EXTRA_MODELS=json.dumps({"en": str(path), "ml": str(path)}),
                      VEYA_ONNX_PATHS=json.dumps({"en": str(path / "veya.onnx"), "ml": str(path / "veya.onnx")}),
                      VEYA_MAX_LOADED="1")
    sys.modules["onnxruntime"] = fake_ort()
    from veya.mcp.tools import ToolError, veya_predict_batch, veya_shortlist, veya_status, veya_tournament

    assert veya_status()["backend"] == "onnx"
    router = serve.build_router()
    assert veya_status(router=router)["device"] == "cpu"
    first = router.predict("refund", QUESTIONS, model="en")
    agent = router.load("en")
    assert agent.session.options.intra_op_num_threads == 2
    calls = agent.session.calls
    assert router.predict("refund", QUESTIONS, model="en")["answers"] == first["answers"]
    assert agent.session.calls == calls
    assert router._veya_decision_cache.cache_info()["hits"] > 0
    router.load("ml")
    assert router.loaded == ["multilingual"]
    router.unload("ml")
    assert router.loaded == []
    assert len(veya_predict_batch([{"state": {"text": "refund"}, "questions": QUESTIONS, "model": "en"}],
                                  router=router)["requests"]) == 1
    large = {"q": {**QUESTIONS["q"], "criteria": {label: label for label in ["a", "b", "c", "d", "e"]}}}
    assert veya_tournament({"text": "refund"}, large, model="en", group_size=2, router=router)["answers"]
    with pytest.raises(ToolError, match="ONNX graph does not export"):
        veya_shortlist({"text": "refund"}, QUESTIONS, model="en", router=router)
    if importlib.util.find_spec("fastapi") is not None:
        app = serve.create_app(router=router, backend="onnx")
        health = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/health")
        assert health(authorization=None)["backend"] == "onnx"
    if importlib.util.find_spec("mcp") is not None:
        import veya.mcp.server as server
        server._ROUTER = None
        server._BACKEND = "onnx"
        local = server._ensure_router()
        assert local.backend == "onnx"
        local.preload(["en"])
        assert json.loads(server.veya_status_tool())["checkpoint_devices"] == {"english": "cpu"}
        local.unload()
        serve._close_decision_cache(local)
        server._ROUTER = None
        os.environ.update(VEYA_BASE_URL="http://127.0.0.1:1", VEYA_BACKEND="invalid", VEYA_ONNX_PATHS="invalid")
        remote = server._ensure_router()
        assert remote.base_url == "http://127.0.0.1:1"
    router.unload()
    serve._close_decision_cache(router)
    assert not any(name.split(".")[0] in ("torch", "transformers") for name in sys.modules)


def test_native_service_lifecycle_never_imports_torch_or_transformers(checkpoint):
    code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['exercise_no_heavy_imports'](Path(sys.argv[2]))"
    run = subprocess.run([sys.executable, "-c", code, __file__, str(checkpoint)],
                         cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
