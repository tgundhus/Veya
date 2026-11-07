"""Exercise installed HTTP and MCP stdio services with real offline ONNX inference."""
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
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class HeavyImportForbidden(BaseException):
    """Optional-import exception handlers must not conceal forbidden imports."""


class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "transformers", "veya"):
            raise HeavyImportForbidden("light service imported " + fullname)


sys.meta_path.insert(0, NoHeavyImports())

FIXTURE = Path(sys.argv[1]).resolve()
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "billing", "support": "help"}, "option_order": [1, 0]},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "high"]},
}
STATE = {"message": "hello billing help"}
PAYLOAD = {"state": STATE, "questions": QUESTIONS, "model": "english"}


def service_environment(cache):
    env = {key: value for key, value in os.environ.items() if not key.startswith("VEYA_")}
    env.pop("PYTHONPATH", None)
    env.update({"VEYA_PRELOAD": "0", "VEYA_MODELS": "english", "VEYA_DEVICE": " CPU ",
                "VEYA_MAX_LOADED": "1", "VEYA_EXTRA_MODELS": json.dumps({
                    "english": str(FIXTURE), "multilingual": str(FIXTURE)}),
                "VEYA_ONNX_PATHS": json.dumps({name: str(FIXTURE / "model.onnx")
                                               for name in ("english", "multilingual")}),
                "VEYA_CACHE_PATH": str(cache), "VEYA_CACHE_TTL": "none",
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    return env


def assert_answers(result):
    answers = result["answers"]
    assert answers["team"]["choice"] == "support", result
    assert set(answers["team"]["probabilities"]) == {"billing", "support"}, result
    assert isinstance(answers["urgent"]["noul"], float), result
    assert isinstance(answers["severity"]["score"], float), result
    assert result["routing"]["model"] == "english", result


def assert_status(status, loaded):
    assert status["loaded"] == loaded, status
    assert status["backend"] == "onnx", status
    assert status["device"] == "cpu", status
    assert status["device_is_preference"] is (not loaded), status
    assert status["checkpoint_devices"] == {name: "cpu" for name in loaded}, status
    assert status["performance_mode"] == 1, status
    assert status["decision_cache"]["enabled"], status


def http_contract(cache):
    env = service_environment(cache)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env.update({"VEYA_HOST": "127.0.0.1", "VEYA_PORT": str(port),
                "VEYA_API_KEY": "offline-contract-key", "VEYA_LOG_LEVEL": "warning",
                "VEYA_IDLE_UNLOAD_SECONDS": "0.8"})
    base = "http://127.0.0.1:%s" % port

    def request(path, payload=None, *, auth=True):
        headers = {"Authorization": "Bearer offline-contract-key"} if auth else {}
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            with urlopen(Request(base + path, data=data, headers=headers), timeout=15) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), str(FIXTURE),
                                    "--http-server"], env=env, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    code, health = request("/health")
                    assert code == 200, health
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        raise AssertionError("HTTP service did not start:\n" + log.read())
                    time.sleep(0.05)
            assert_status(health, [])
            assert request("/health", auth=False) == (200, {"status": "ok"})
            assert request("/v1/systemone", PAYLOAD, auth=False)[0] == 401
            assert request("/v1/systemone", {**PAYLOAD, "max_len": True})[0] == 422
            assert request("/v1/systemone", {**PAYLOAD, "model": "unknown/checkpoint"})[0] == 422
            code, answer = request("/v1/systemone", PAYLOAD)
            assert code == 200, answer
            assert_answers(answer)
            _, repeated = request("/v1/systemone", PAYLOAD)
            assert repeated["answers"] == answer["answers"]
            _, health = request("/health")
            assert_status(health, ["english"])
            assert health["decision_cache"]["hits"] >= 1, health
            code, batch = request("/v1/systemone/batch", {
                "states": [STATE, {"message": "billing"}], "questions": QUESTIONS, "model": "english"})
            assert code == 200, batch
            assert len(batch["results"]) == 2, batch
            for result in batch["results"]:
                assert_answers(result)
            # One resident graph: another checkpoint evicts the first without Torch cleanup.
            code, other = request("/v1/systemone", {**PAYLOAD, "state": {"message": "other"},
                                                    "model": "multilingual"})
            assert code == 200, other
            assert_status(request("/health")[1], ["multilingual"])
            deadline = time.monotonic() + 8
            while request("/health")[1]["loaded"]:
                assert time.monotonic() < deadline, "idle graph did not unload"
                time.sleep(0.1)
            unloaded = request("/health")[1]
            assert_status(unloaded, [])
            # Retention survives unloading. Router loads the checkpoint before its
            # cache hook runs, then replays the answer without a graph forward pass.
            assert_answers(request("/v1/systemone", PAYLOAD)[1])
            replayed = request("/health")[1]
            assert_status(replayed, ["english"])
            assert replayed["decision_cache"]["hits"] > unloaded["decision_cache"]["hits"], replayed
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


async def mcp_contract(cache):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable,
        args=["-I", str(Path(__file__).resolve()), str(FIXTURE), "--mcp-server"],
        env=service_environment(cache), cwd=str(FIXTURE.parent))
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=20) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {"veya_predict", "veya_predict_batch", "veya_route", "veya_status", "veya_preset",
                    "veya_decide", "veya_predict_long", "veya_shortlist", "veya_tournament"} <= {
                        tool.name for tool in listed.tools}

            async def call(name, arguments=None, *, error=False):
                result = await session.call_tool(name, arguments or {})
                # SDK Python attributes differ across versions; protocol aliases stay stable.
                wire = result.model_dump(mode="json", by_alias=True)
                assert bool(wire.get("isError", False)) is error, wire
                text = "".join(part["text"] for part in wire["content"] if part["type"] == "text")
                # MCP can prepend framework context to a tool error's JSON message.
                return {"message": text} if error else json.loads(text)

            status = await call("veya_status")
            assert status["backend"] == "onnx" and status["loaded"] == [], status
            assert status["device"] == "cpu" and status["device_is_preference"], status
            route = await call("veya_route", PAYLOAD)
            assert route["model"] == "english", route
            assert_status(await call("veya_status"), [])
            invalid = await call("veya_predict", {**PAYLOAD, "max_len": -1}, error=True)
            assert "invalid_max_len" in invalid["message"], invalid
            first = await call("veya_predict", PAYLOAD)
            assert_answers(first)
            assert first["device"] == "cpu", first
            assert (await call("veya_predict", PAYLOAD))["answers"] == first["answers"]
            status = await call("veya_status")
            assert_status(status, ["english"])
            assert status["decision_cache"]["hits"] >= 1, status
            batch = await call("veya_predict_batch", {"requests": [PAYLOAD, PAYLOAD]})
            assert len(batch["requests"]) == 2, batch
            for result in batch["requests"]:
                assert_answers(result)
            decided = await call("veya_decide", {"state": STATE, "model": "english", "schema": {
                "type": "object", "properties": {"urgent": {"type": "boolean"}}}})
            assert isinstance(decided["values"]["urgent"], bool), decided
            preset = await call("veya_preset", {"preset": "guard", "state": STATE, "lang": "en"})
            assert preset["answers"], preset
            scanned = await call("veya_predict_long", {**PAYLOAD,
                "state": {"message": "hello billing help " * 30}, "window": 16, "stride": 8})
            assert scanned["windows"] > 1, scanned
            assert all("window" in answer for answer in scanned["answers"].values()), scanned
            tournament = await call("veya_tournament", {"state": STATE, "model": "english", "group_size": 2,
                "questions": {"team": {"type": "choice", "instructions": "Which team?",
                                       "criteria": {str(i): "billing help" for i in range(5)}}}})
            finalists = tournament["tournament"]["team"]
            assert finalists["n"] == 5 and finalists["rounds"] > 0, tournament
            assert tournament["answers"]["team"]["choice"] in finalists["labels"], tournament
            assert tournament["device"] == "cpu", tournament
            # Encoder embeddings are absent from exported decision graphs: reject clearly.
            unsupported = await call("veya_shortlist", {**PAYLOAD, "k": 1}, error=True)
            assert "ONNX" in unsupported["message"], unsupported
            await call("veya_predict", {**PAYLOAD, "state": {"message": "other"}, "model": "multilingual"})
            assert_status(await call("veya_status"), ["multilingual"])


def main():
    for name in ("torch", "transformers", "veya", "onnx", "onnxscript", "safetensors"):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError("service environment contains " + name)
    with tempfile.TemporaryDirectory() as tmp:
        http_contract(Path(tmp) / "http.sqlite")
        asyncio.run(mcp_contract(Path(tmp) / "mcp.sqlite"))
    assert not any(name.split(".")[0] in ("torch", "transformers", "veya") for name in sys.modules)
    print("PASS installed HTTP and MCP stdio protocols, native ONNX inference and service lifecycle")


if __name__ == "__main__":
    if "--http-server" in sys.argv:
        from veya_onnx.serve import main as serve_main

        def stop_on_parent_exit():
            # Invoke uvicorn's installed shutdown handler after stdin closes. Windows
            # terminate() kills the venv launcher before its interpreter releases SQLite.
            sys.stdin.buffer.read()
            handler = signal.getsignal(signal.SIGTERM)
            if callable(handler):
                handler(signal.SIGTERM, None)

        threading.Thread(target=stop_on_parent_exit, daemon=True).start()
        serve_main(["--performance-mode", "1"])
    elif "--mcp-server" in sys.argv:
        from veya_onnx.mcp.server import main as mcp_main

        mcp_main(["--performance-mode", "1"])
    else:
        main()
