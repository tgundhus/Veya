"""Verify installed local code execution and cold replay with heavy imports forbidden."""
import asyncio
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen

if "--prepare-only" in sys.argv:
    # Import the artifact under test directly as a zip, never the source checkout.
    sys.path.insert(0, str(Path(sys.argv[3]).resolve()))


class HeavyImportForbidden(BaseException):
    """Keep forbidden source-package imports visible across optional fallbacks."""


class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {"torch", "transformers", "laya"}
        if "--cold-replay" in sys.argv or "--http-replay-server" in sys.argv:
            forbidden.update(("onnx", "onnxruntime", "tokenizers"))
        if fullname.split(".")[0] in forbidden:
            raise HeavyImportForbidden("production runtime imported " + fullname)


sys.meta_path.insert(0, NoHeavyImports())

FIXTURE = Path(sys.argv[1]).resolve()
QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": "billing", "support": "help"}, "option_order": [1, 0]},
             "urgent": {"type": "noul", "instructions": "Is it urgent?"}}
STATE = {"message": "hello billing help"}
PAYLOAD = {"state": STATE, "questions": QUESTIONS, "model": "english"}
POLICY = "installed-production-contract-v1"


def router_options():
    return {"models": {name: str(FIXTURE) for name in ("english", "multilingual")},
            "onnx_paths": {name: str(FIXTURE / "model.onnx") for name in ("english", "multilingual")},
            "backend": "onnx", "device": "cpu", "performance_mode": 1, "max_loaded": 1}


def environment(cache, manifest=None, store_id=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("LAYA_")}
    env.pop("PYTHONPATH", None)
    options = router_options()
    env.update(LAYA_PRELOAD="0", LAYA_MODELS="english", LAYA_DEVICE="cpu", LAYA_MAX_LOADED="1",
               LAYA_EXTRA_MODELS=json.dumps(options["models"]), LAYA_ONNX_PATHS=json.dumps(options["onnx_paths"]),
               LAYA_CACHE_PATH=str(cache), LAYA_CACHE_TTL="none", LAYA_CACHE_MAXSIZE="none",
               LAYA_CACHE_REQUIRED="1", LAYA_CACHE_FINGERPRINT=POLICY,
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if manifest is not None:
        env.update(LAYA_REPLAY_MANIFEST=str(manifest), LAYA_CACHE_REQUIRE_EXISTING="1",
                   LAYA_CACHE_EXPECTED_STORE_ID=store_id)
    return env


def cold_replay(cache_path, manifest, store_id):
    from laya_onnx import DecisionCache
    from laya_onnx._runtime.router import Router

    assert not {"onnx", "onnxruntime", "tokenizers"} & set(sys.modules), "cold replay imported model tooling"
    cache = DecisionCache(str(cache_path), required=True, fingerprint=POLICY,
                          require_existing=True, expected_store_id=store_id, maxsize=None)
    try:
        router = Router(**router_options(), hooks=[cache], replay_manifest=str(manifest))

        def no_load(*args, **kwargs):
            raise AssertionError("verified replay attempted to load weights")

        router.load = no_load
        first = router.try_replay(STATE, QUESTIONS, model="english")
        assert first is not None
        assert first["answers"]["team"]["choice"] == "support", first
        assert router.predict(STATE, QUESTIONS, model="english")["answers"] == first["answers"]
        assert router.try_replay({"message": "a distinct uncached input"}, QUESTIONS, model="english") is None
        assert not router._agents and not {"onnx", "onnxruntime", "tokenizers"} & set(sys.modules)
        print(json.dumps({"answers": first["answers"], "loaded": list(router._agents)}))
    finally:
        cache.close()


def prepare_replay(root):
    from laya_onnx import DecisionCache, DecisionCacheError
    from laya_onnx._runtime.router import Router

    cache_path, manifest = root / "decisions.sqlite", root / "replay.json"
    cache = DecisionCache(str(cache_path), required=True, fingerprint=POLICY, maxsize=None)
    router = Router(**router_options(), hooks=[cache])
    try:
        first = router.predict(STATE, QUESTIONS, model="english")
        assert first["answers"]["team"]["choice"] == "support", first
        identity = cache.store_id
        report = router.export_replay_manifest(str(manifest))
        assert report["store_id"] == identity and report["models"] == ["english"], report
        assert manifest.is_file()
    finally:
        router.unload()
        cache.close()
    with_missing = root / "missing.sqlite"
    try:
        DecisionCache(str(with_missing), required=True, require_existing=True, expected_store_id=identity)
    except DecisionCacheError:
        pass
    else:
        raise AssertionError("required existing store created a missing database")
    assert not with_missing.exists()
    try:
        DecisionCache(str(cache_path), required=True, expected_store_id="00000000-0000-0000-0000-000000000000")
    except DecisionCacheError:
        pass
    else:
        raise AssertionError("a different decision-store identity was accepted")
    return cache_path, manifest, identity, first["answers"]


def local_code_contract(cache_path, manifest, store_id, expected):
    from laya_onnx import DecisionCache
    from laya_onnx.mcp.code_mode import CodeModeError, CodeModeLimits, CodeModeRunner
    from laya_onnx._runtime.router import Router

    cache = DecisionCache(str(cache_path), required=True, fingerprint=POLICY, maxsize=None,
                          require_existing=True, expected_store_id=store_id)
    router = Router(**router_options(), hooks=[cache], replay_manifest=str(manifest))
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=15))
    try:
        empty = runner.execute("async (input) => ({value: input.value * 2, ambient: "
                               "[typeof process, typeof require, typeof fetch, typeof std, typeof os]})",
                               {"value": 21})
        assert empty == {"value": 42, "ambient": ["undefined"] * 5}, empty
        code = "async (input, veya) => (await veya.predict(input)).answers"
        assert runner.execute(code, PAYLOAD, router=router) == expected
        assert not router._agents, "code-mode replay loaded a model"
        try:
            runner.execute("async () => { while (true) {} }", router=router)
        except CodeModeError as error:
            assert error.code in ("code_script_failed", "code_cpu_limit", "code_timeout"), error
        else:
            raise AssertionError("an infinite script escaped its CPU/wall limit")
        try:
            runner.execute("(() => { throw {toString() { while (true) {} }}; })()", router=router)
        except CodeModeError as error:
            assert error.code in ("code_script_failed", "code_cpu_limit", "code_timeout"), error
        else:
            raise AssertionError("guest exception conversion escaped the sandbox limits")
        cache.close()
        try:
            runner.execute("async (input, veya) => { try { return await veya.predict(input); } "
                           "catch (_) { return {alternative: true}; } }", PAYLOAD, router=router)
        except CodeModeError as error:
            assert error.code == "decision_cache_unavailable", error
        else:
            raise AssertionError("JavaScript caught a required-cache failure and returned an alternative")
    finally:
        runner.close()
        router.unload()
        cache.close()


def public_agent_code_contract(expected):
    from laya_onnx import DecisionCache, ONNXAgent
    from laya_onnx.mcp.code_mode import CodeModeRunner
    from laya_onnx._runtime.router import Router

    agent = ONNXAgent(str(FIXTURE), str(FIXTURE / "model.onnx"), performance_mode=1)
    cache = DecisionCache(required=True, fingerprint=POLICY, maxsize=None)
    router = Router(backend="onnx", performance_mode=1, hooks=[cache])
    router.attach("english", agent)
    runner = CodeModeRunner()
    try:
        code = "async (input, veya) => (await veya.predict(input)).answers"
        assert runner.execute(code, PAYLOAD, router=router) == expected
        assert runner.execute(code, PAYLOAD, router=router) == expected
        assert cache.cache_info()["hits"] >= 1
    finally:
        runner.close()
        router.unload()
        cache.close()


def public_decision_policy_contract():
    from laya_onnx import ONNXAgent
    from laya_onnx.mcp.code_mode import CodeModeError, CodeModeLimits, CodeModeRunner
    from laya_onnx._runtime.router import Router

    questions = dict(QUESTIONS, priority={"type": "score", "instructions": "Rate priority",
                                         "criteria": ["low", "medium", "high"]})
    agent = ONNXAgent(str(FIXTURE), str(FIXTURE / "model.onnx"), performance_mode=1,
                      decision_policy="context-balanced-v1")
    router = Router(backend="onnx", performance_mode=1)
    router.attach("english", agent)
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=15, max_batch_tokens=512))
    try:
        answer = agent.predict(STATE, questions)
        assert agent.decision_policy == "context-balanced-v1"
        # This graph depends only on marker positions, not the state. Removing its
        # presentation/content-free prior must therefore produce uniform answers.
        assert all(abs(value - .5) < 1e-4 for value in answer["answers"]["team"]["probabilities"].values()), answer
        assert answer["answers"]["urgent"]["noul"] == .5, answer
        assert answer["answers"]["priority"]["score"] == 1.0, answer
        assert all(row["decision_policy"] == "context-balanced-v1" and row["confidence_calibrated"] is False
                   and "_decision_probabilities" not in row for row in answer["answers"].values()), answer
        assert answer["usage"]["policy_variant_rows"] == 7, answer
        assert answer["usage"]["policy_probe_rows"] == 21, answer
        assert answer["usage"]["policy_probe_tokens"] > 0, answer
        batch = agent.predict_batch([STATE, STATE], questions, batch_size=1)
        assert [row["answers"] for row in batch] == [answer["answers"], answer["answers"]], batch
        assert sum(row["usage"]["policy_probe_rows"] for row in batch) == 21, batch

        class CountSession:
            def __init__(self, session):
                self.session, self.calls = session, 0

            def __getattr__(self, name):
                return getattr(self.session, name)

            def run(self, *args, **kwargs):
                self.calls += 1
                return self.session.run(*args, **kwargs)

        agent.session = CountSession(agent.session)
        try:
            runner.execute("async input => await veya.predict(input)",
                           dict(PAYLOAD, questions=questions), router=router)
        except CodeModeError as error:
            assert error.code == "code_resource_limit", error
        else:
            raise AssertionError("Code Mode ignored balanced inference/probe work")
        assert agent.session.calls == 0, "oversized policy work reached an ONNX forward"
    finally:
        runner.close()
        router.unload()


def http_replay_contract(cache_path, manifest, store_id, expected):
    env = environment(cache_path, manifest, store_id)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env.update(LAYA_HOST="127.0.0.1", LAYA_PORT=str(port), LAYA_LOG_LEVEL="warning",
               LAYA_API_KEY="installed-replay-contract", LAYA_MICROBATCH_MAX_STATES="8",
               LAYA_MICROBATCH_WAIT_MS="200")
    base = "http://127.0.0.1:%d" % port

    def request(path, payload=None):
        headers = {"Authorization": "Bearer installed-replay-contract"}
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        with urlopen(Request(base + path, data=data, headers=headers), timeout=15) as response:
            return json.load(response), response.headers

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), str(FIXTURE),
                                    "--http-replay-server"], env=env, stdin=subprocess.PIPE,
                                   stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    health, _ = request("/health")
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise AssertionError("HTTP replay service did not start:\n" + log.read())
                    time.sleep(0.05)
            assert health["loaded"] == [], health
            assert health["model_free_replay"]["enabled"] and health["model_free_replay"]["http_fast_path"], health
            answer, headers = request("/v1/systemone", PAYLOAD)
            assert {name: {key: value for key, value in row.items() if key != "x_jev_confidence"}
                    for name, row in answer["answers"].items()} == expected, answer
            assert headers["X-Decision-Cache"] == "hit", headers
            assert headers["X-Inference-Time-Ms"] == "0.00", headers
            assert ", queue;" not in headers["Server-Timing"] and "cache;dur=" in headers["Server-Timing"], headers
            assert request("/health")[0]["loaded"] == []
        except BaseException:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            if process.poll() is None:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


async def mcp_code_contract(cache_path, manifest, store_id, expected):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable,
        args=["-I", str(Path(__file__).resolve()), str(FIXTURE), "--mcp-code-server"],
        env=environment(cache_path, manifest, store_id), cwd=str(FIXTURE.parent))
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=30) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert [tool.name for tool in listed.tools] == ["veya_execute"], listed

            async def call(code, input):
                result = await session.call_tool("veya_execute", {"code": code, "input": input})
                wire = result.model_dump(mode="json", by_alias=True)
                assert not wire.get("isError", False), wire
                text = "".join(part["text"] for part in wire["content"] if part["type"] == "text")
                return json.loads(text)

            code = "async (input, veya) => (await veya.predict(input)).answers"
            assert await call(code, PAYLOAD) == expected
            status = await call("async (_, veya) => await veya.status({})", None)
            assert status["backend"] == "onnx" and status["loaded"] == [], status
            assert status["decision_cache"]["required"], status
            batch = await call("async (input, veya) => { const result = await veya.predict_batch({requests: input}); "
                               "return result.requests.map(item => item.answers); }", [PAYLOAD, PAYLOAD])
            assert batch == [expected, expected], batch


def main():
    for name in ("torch", "transformers", "veya", "onnx", "onnxscript", "safetensors"):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError("production environment contains " + name)
    assert importlib.metadata.version("quickjs-ng")
    prepared = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    cache_path, manifest = Path(prepared["cache"]), Path(prepared["manifest"])
    store_id, expected = prepared["store_id"], prepared["answers"]
    program = [sys.executable, "-I", str(Path(__file__).resolve()), str(FIXTURE), "--cold-replay",
               str(cache_path), str(manifest), store_id]
    output = subprocess.check_output(program, cwd=cache_path.parent, env=environment(cache_path),
                                     text=True, timeout=30)
    replayed = json.loads(output)
    assert replayed["answers"] == expected and replayed["loaded"] == [], replayed
    http_replay_contract(cache_path, manifest, store_id, expected)
    local_code_contract(cache_path, manifest, store_id, expected)
    public_agent_code_contract(expected)
    public_decision_policy_contract()
    asyncio.run(mcp_code_contract(cache_path, manifest, store_id, expected))
    assert not any(name.split(".")[0] in ("torch", "transformers", "laya") for name in sys.modules)
    print("PASS installed required store identity, model-free replay and sandboxed local MCP code")


if __name__ == "__main__":
    if "--prepare-only" in sys.argv:
        destination = Path(sys.argv[4]).resolve()
        destination.mkdir()
        cache, manifest, identity, answers = prepare_replay(destination)
        (destination / "contract.json").write_text(json.dumps({"cache": str(cache), "manifest": str(manifest),
            "store_id": identity, "answers": answers}), encoding="utf-8")
    elif "--cold-replay" in sys.argv:
        cold_replay(Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
    elif "--http-replay-server" in sys.argv:
        from laya_onnx.serve import main as serve_main

        def stop_on_parent_exit():
            sys.stdin.buffer.read()
            handler = signal.getsignal(signal.SIGTERM)
            if callable(handler):
                handler(signal.SIGTERM, None)

        threading.Thread(target=stop_on_parent_exit, daemon=True).start()
        serve_main(["--performance-mode", "1"])
    elif "--mcp-code-server" in sys.argv:
        from laya_onnx.mcp.server import main as mcp_main

        mcp_main(["--performance-mode", "1", "--mode", "code"])
    else:
        main()
