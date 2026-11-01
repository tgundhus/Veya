"""Offline real-transport Torch/native-ONNX service comparison.

Run --self-check without checkpoints. Normal runs use three paired fresh processes
per transport. The same interpreter/dependencies serve the frozen main Torch path,
current full-package ONNX path, and a code-only extraction of the standalone wheel.
The latter controls dependency versions; isolated installation is checked separately
by runtime/onnx/tests/verify_package.py. No model is downloaded or exported here.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import functools
import http.client
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import queue
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import zipfile

from bench_loading_memory import QUESTIONS, STATES, baseline_tree, sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = "e823ea169941193b83b91322ecd01b1d21f32a83"
VARIANTS = ("baseline-torch", "full-onnx", "lite-onnx")
PUBLIC_TOLERANCE = 0.0002
LOGIT_TOLERANCE = 0.0001
PROBABILITY_TOLERANCE = 0.00002
STATE_OBJECTS = [{"body": value} for value in STATES]
CONTROLS = {"model": "en", "max_len": 192, "head_max_len": 96}


def process_memory(pid):
    """Read the server's OS-maintained lifetime peaks, without a polling sampler."""
    if os.name != "nt":
        raise RuntimeError("This benchmark's memory report requires Windows")
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                "PrivateUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"working_set_bytes": counters.WorkingSetSize, "private_commit_bytes": counters.PrivateUsage,
                "peak_working_set_bytes": counters.PeakWorkingSetSize,
                "peak_commit_bytes": counters.PeakPagefileUsage, "page_faults": counters.PageFaultCount,
                "source": "Windows GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX"}
    finally:
        kernel.CloseHandle(handle)


def extract_wheel(wheel, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Wheel member escapes extraction directory")
        archive.extractall(destination)
    return destination


def numeric_difference(left, right, path=""):
    """Compare public answer shape/labels exactly and finite numbers by absolute delta."""
    if isinstance(left, bool) or isinstance(right, bool):
        assert left == right, (path, left, right)
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        assert math.isfinite(left) and math.isfinite(right), path
        return abs(left - right)
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), (path, set(left), set(right))
        return max([numeric_difference(left[key], right[key], path + "/" + str(key))
                    for key in left] or [0.0])
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), path
        return max([numeric_difference(a, b, path + "/" + str(i))
                    for i, (a, b) in enumerate(zip(left, right))] or [0.0])
    assert left == right, (path, left, right)
    return 0.0


def canonical_answers(response, transport, kind):
    entries = ([response] if kind == "single" else
               response["results" if transport == "http" else "requests"])
    answers = []
    for entry in entries:
        assert entry.get("routing", {}).get("model") == "english", entry
        answers.append({qid: {key: value for key, value in answer.items() if key != "x_jev_confidence"}
                        for qid, answer in entry["answers"].items()})
    return answers


def record_difference(left, right):
    assert len(left) == len(right), (len(left), len(right))
    output = {}
    for field in ("logits", "probabilities", "action_probabilities"):
        values = []
        for a, b in zip(left, right):
            assert a["question"] == b["question"]
            assert len(a[field]) == len(b[field])
            assert all(math.isfinite(value) for value in a[field] + b[field])
            values.extend(abs(x - y) for x, y in zip(a[field], b[field]))
        output["max_abs_" + field] = max(values or [0.0])
    return output


def child(config):
    """Launch the actual service main, adding measurement wrappers outside model work."""
    sys.path.insert(0, config["source"])
    prefix = "laya_onnx._runtime" if config["variant"] == "lite-onnx" else "laya"
    router_module = importlib.import_module(prefix + ".router")
    router_class = router_module.Router
    state = {"router": None, "calls": [], "records": [], "capture": False}
    original_build = router_class._build

    @functools.wraps(original_build)
    def build(router, *args, **kwargs):
        agent = original_build(router, *args, **kwargs)
        decode = agent._decode_answers

        @functools.wraps(decode)
        def capture(logits, act, items, ids, internal, offset, lang=None):
            if state["capture"]:
                import numpy as np
                for j, qid in enumerate(ids):
                    kind = {"choice": 0, "score": 1, "noul": 2}[internal[qid]["t"]]
                    count = len(items[j]["markers"])
                    row = np.array(logits[offset + j, :count], copy=True)
                    bucket = "%s:%s" % (internal[qid]["t"], "2" if count == 2 else
                                        "3-5" if count <= 5 else "6-10" if count <= 10 else "11+")
                    temperature = agent.temperature_by_options.get(bucket, agent.temperature[kind])
                    values = row / temperature
                    probs = np.exp(values - values.max())
                    probs /= probs.sum()
                    state["records"].append({"question": qid, "logits": row.tolist(),
                                             "probabilities": probs.tolist(),
                                             "action_probabilities": act[offset + j].tolist()})
            return decode(logits, act, items, ids, internal, offset, lang)

        agent._decode_answers = capture
        return agent

    router_class._build = build
    for name in ("predict", "predict_batch"):
        original = getattr(router_class, name)

        def measured(router, *args, _original=original, _name=name, **kwargs):
            state["router"] = router
            started = time.perf_counter()
            result = _original(router, *args, **kwargs)
            state["calls"].append({"method": _name, "elapsed_ms": (time.perf_counter() - started) * 1000})
            return result

        setattr(router_class, name, measured)

    def details():
        router = state["router"]
        agents = []
        for name, agent in router._agents.items():
            item = {"name": name, "class": type(agent).__module__ + "." + type(agent).__name__,
                    "tokenizer_class": type(agent.tok).__module__ + "." + type(agent.tok).__name__,
                    "performance_mode": agent.performance_mode}
            if hasattr(agent, "session"):
                item.update(providers=agent.session.get_providers(),
                            threads=agent.session.get_session_options().intra_op_num_threads,
                            output_types=[output.type for output in agent.session.get_outputs()])
                assert item["providers"] == ["CPUExecutionProvider"]
                assert item["threads"] == config["threads"]
                assert item["output_types"] == ["tensor(float)", "tensor(float)"]
            else:
                import torch
                item.update(dtype=str(agent.dtype), amp_enabled=agent.amp_enabled,
                            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                            backend=getattr(agent, "backend", None))
                assert item["dtype"] == "torch.float32" and not item["amp_enabled"]
                assert item["threads"] == config["threads"]
            agents.append(item)
        assert [agent["name"] for agent in agents] == ["english"]
        packages = {}
        for name in ("numpy", "tokenizers", "onnxruntime", "torch", "transformers", "fastapi", "uvicorn", "mcp"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        executed = {}
        package_prefix = "laya_onnx" if config["variant"] == "lite-onnx" else "laya"
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if (name == package_prefix or name.startswith(package_prefix + ".")) and path and path.endswith(".py"):
                executed[str(Path(path).resolve())] = sha256(Path(path))
        cache = getattr(router, "_laya_decision_cache", None)
        return {"calls": list(state["calls"]), "records": list(state["records"]), "agents": agents,
                "packages": packages, "executed_source_sha256": executed,
                "imported_torch": "torch" in sys.modules, "imported_transformers": "transformers" in sys.modules,
                "cache_info": cache.cache_info() if cache else None}

    def control_loop():
        last = None
        request = Path(config["control"])
        response = Path(config["ack"])
        while True:
            try:
                command = json.loads(request.read_text(encoding="utf-8"))
                if command["id"] != last:
                    action = command["action"]
                    if action == "pid":
                        payload = {"server_pid": os.getpid()}
                    elif action == "exit":
                        response.write_text(json.dumps({"id": command["id"], "payload": {
                            "server_pid": os.getpid(), "exiting": True}}), encoding="utf-8")
                        os._exit(0)
                    elif action == "capture":
                        state["capture"] = True
                        payload = {"capture": True}
                    elif action == "enable-cache":
                        os.environ["LAYA_CACHE"] = "1"
                        importlib.import_module(prefix + ".serve")._configure_decision_cache(state["router"])
                        payload = {"enabled": True}
                    else:
                        payload = details()
                    payload["server_pid"] = os.getpid()
                    response.write_text(json.dumps({"id": command["id"], "payload": payload}), encoding="utf-8")
                    last = command["id"]
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            except Exception as exc:
                response.write_text(json.dumps({"id": command["id"], "error": repr(exc)}), encoding="utf-8")
                last = command["id"]
            time.sleep(0.02)

    threading.Thread(target=control_loop, daemon=True).start()
    module_name = ("laya_onnx" if config["variant"] == "lite-onnx" else "laya")
    module_name += ".serve" if config["transport"] == "http" else ".mcp.server"
    cli = ["--performance-mode", "default"]
    if config["variant"] != "baseline-torch":
        cli += ["--backend", "onnx"]
    # Torch inter-op is fixed at one for all baseline processes. ONNX never imports torch.
    if config["variant"] == "baseline-torch":
        import torch
        torch.set_num_interop_threads(1)
    importlib.import_module(module_name).main(cli)


class Client:
    def __init__(self, process, transport, port, timeout):
        self.process, self.transport, self.timeout = process, transport, timeout
        self.next_id = 0
        self.handshake = None
        self.health = None
        if transport == "http":
            deadline = time.perf_counter() + timeout
            while True:
                try:
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    self.health = json.loads(response.read())
                    connection.close()
                    assert response.status == 200
                    break
                except (OSError, http.client.HTTPException):
                    if process.poll() is not None or time.perf_counter() > deadline:
                        raise RuntimeError("HTTP server did not become ready")
                    time.sleep(0.02)
            self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        else:
            self.lines = queue.Queue()

            def drain():
                for line in process.stdout:
                    self.lines.put(line)
                self.lines.put(None)

            threading.Thread(target=drain, daemon=True).start()
            self.handshake = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "laya-onnx-service-benchmark", "version": "1"}})
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            process.stdin.flush()

    def rpc(self, method, params):
        self.next_id += 1
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id,
                                            "method": method, "params": params}) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.lines.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError("MCP server closed stdout")
            response = json.loads(line)
            if response.get("id") == self.next_id:
                assert "error" not in response, response
                return response["result"]

    def predict(self, kind, index=0):
        started = time.perf_counter()
        if self.transport == "http":
            body = dict(questions=QUESTIONS, **CONTROLS)
            body["state" if kind == "single" else "states"] = STATE_OBJECTS[index] if kind == "single" else STATE_OBJECTS
            self.connection.request("POST", "/v1/systemone" + ("/batch" if kind == "batch" else ""),
                                    body=json.dumps(body), headers={"Content-Type": "application/json"})
            response = self.connection.getresponse()
            payload = json.loads(response.read())
            assert response.status == 200, (response.status, payload)
        else:
            arguments = dict(state=STATE_OBJECTS[index], questions=QUESTIONS, **CONTROLS)
            if kind == "batch":
                arguments = {"requests": [dict(state=state, questions=QUESTIONS, **CONTROLS) for state in STATE_OBJECTS]}
            result = self.rpc("tools/call", {"name": "laya_predict" + ("_batch" if kind == "batch" else ""),
                                             "arguments": arguments})
            assert not result.get("isError"), result
            payload = json.loads(result["content"][0]["text"])
        return {"kind": kind, "elapsed_ms": (time.perf_counter() - started) * 1000, "response": payload}


def control(config, action, sequence):
    Path(config["control"]).write_text(json.dumps({"id": sequence, "action": action}), encoding="utf-8")
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        try:
            response = json.loads(Path(config["ack"]).read_text(encoding="utf-8"))
            if response["id"] == sequence:
                assert "error" not in response, response
                return response["payload"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    raise TimeoutError("Measurement control did not acknowledge " + action)


def run_service(args, variant, transport, repeat, source, workspace, env):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    identity = "%s-%s-%d" % (transport, variant, repeat)
    config = {"variant": variant, "transport": transport, "source": str(source), "threads": args.threads,
              "control": str(workspace / (identity + ".control.json")),
              "ack": str(workspace / (identity + ".ack.json"))}
    config_path = workspace / (identity + ".config.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for key in ("control", "ack"):
        Path(config[key]).unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--child-config", str(config_path)]
    process_env = dict(env, LAYA_PORT=str(port))
    stderr = []
    launched = time.perf_counter()
    process = subprocess.Popen(command, cwd=workspace, env=process_env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    def drain_stderr():
        for line in process.stderr:
            stderr.append(line.rstrip("\n"))

    thread = threading.Thread(target=drain_stderr, daemon=True)
    thread.start()
    row = {"variant": variant, "transport": transport, "process_repeat": repeat, "source": str(source),
           "command": command, "launcher_pid": process.pid, "requests": [], "memory_stages": {}}
    try:
        # Windows venv python.exe can be a redirector process. Query the actual
        # interpreter before readiness; memory and cleanup must target that PID.
        row["server_pid"] = control(config, "pid", 0)["server_pid"]
        client = Client(process, transport, port, args.timeout)
        ready = time.perf_counter()
        row.update(ready_s=ready - launched, handshake=client.handshake, health=client.health)
        first = client.predict("single")
        answered = time.perf_counter()
        first["phase"] = "first"
        row["requests"].append(first)
        row["startup_to_first_answer_s"] = answered - launched
        row["memory_stages"]["after_first_answer"] = process_memory(row["server_pid"])
        for kind in ("single", "batch"):
            for _ in range(args.warmups):
                request = client.predict(kind)
                request["phase"] = "warmup"
                row["requests"].append(request)
            for _ in range(args.repeats):
                request = client.predict(kind)
                request["phase"] = "warm"
                row["requests"].append(request)
        row["memory_stages"]["after_cache_off_timings"] = process_memory(row["server_pid"])
        control(config, "capture", 1)
        for index in range(len(STATE_OBJECTS)):
            request = client.predict("single", index)
            request["phase"] = "parity"
            row["requests"].append(request)
        request = client.predict("batch")
        request["phase"] = "parity"
        row["requests"].append(request)
        before_cache = control(config, "report", 2)
        assert before_cache["cache_info"] is None
        row["cache_off_call_count"] = len(before_cache["calls"])
        row["records"] = before_cache["records"]
        control(config, "enable-cache", 3)
        for phase, count in (("cache_fill", 1), ("cache_hit", args.cache_repeats)):
            for _ in range(count):
                request = client.predict("single")
                request["phase"] = phase
                row["requests"].append(request)
        row["measurement"] = control(config, "report", 4)
        cache = row["measurement"]["cache_info"]
        assert cache["hits"] == args.cache_repeats and cache["misses"] == 1 and cache["errors"] == 0, cache
        assert len(row["measurement"]["calls"]) == len(row["requests"])
        for request, measured in zip(row["requests"], row["measurement"]["calls"]):
            assert measured["method"] == ("predict" if request["kind"] == "single" else "predict_batch")
            request["router_ms"] = measured["elapsed_ms"]
            request["outside_router_ms"] = request["elapsed_ms"] - measured["elapsed_ms"]
        row["memory_stages"]["after_cache_and_parity"] = process_memory(row["server_pid"])
        if variant != "baseline-torch":
            assert not row["measurement"]["imported_torch"]
            assert not row["measurement"]["imported_transformers"]
        return row
    except Exception as exc:
        raise RuntimeError("%s failed; stderr tail: %s" % (identity, "\n".join(stderr[-30:]))) from exc
    finally:
        if process.poll() is None:
            try:
                # End the model-owning interpreter first; killing only the venv
                # launcher can leave a resident server behind on Windows.
                control(config, "exit", 5)
                process.wait(timeout=20)
            except Exception:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False)
                if process.poll() is None:
                    process.kill()
        process.wait(timeout=20)
        thread.join(timeout=5)
        row["stderr"] = stderr
        row["termination_returncode"] = process.returncode


def parity(rows):
    reference = next(row for row in rows if row["variant"] == "baseline-torch")
    expected = [canonical_answers(req["response"], reference["transport"], req["kind"])
                for req in reference["requests"] if req["phase"] == "parity"]
    output = []
    for row in rows:
        actual = [canonical_answers(req["response"], row["transport"], req["kind"])
                  for req in row["requests"] if req["phase"] == "parity"]
        delta = numeric_difference(expected, actual)
        raw = record_difference(reference["records"], row["records"])
        assert delta <= PUBLIC_TOLERANCE, delta
        assert raw["max_abs_logits"] <= LOGIT_TOLERANCE, raw
        assert raw["max_abs_probabilities"] <= PROBABILITY_TOLERANCE, raw
        assert raw["max_abs_action_probabilities"] <= PROBABILITY_TOLERANCE, raw
        output.append({"variant": row["variant"], "transport": row["transport"],
                       "process_repeat": row["process_repeat"], "max_abs_public_answer_number": delta,
                       "labels_and_schema_equal": True, **raw})
    return output


def markdown(report):
    lines = ["# Local ONNX service measurements", "",
             "Real localhost HTTP and MCP stdio; one pinned English checkpoint, fp32 CPU, %d threads. "
             "Cache disabled for first/warm single/batch timings. Filesystem caches were left intact. "
             "All variants use the same interpreter and dependency versions. Lightweight distribution "
             "installation is verified separately; this is a controlled runtime comparison." % report["threads"], "",
             "| transport | service | start to first answer s | first request ms | warm single ms | "
             "warm batch (4 states) ms | cache hit ms | peak WS MiB | peak private commit MiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for transport in report["transports"]:
        for variant in report["variants"]:
            rows = [r for r in report["runs"] if r["transport"] == transport and r["variant"] == variant]
            values = [statistics.median(r["startup_to_first_answer_s"] for r in rows),
                      statistics.median(r["requests"][0]["elapsed_ms"] for r in rows)]
            for phase, kind in (("warm", "single"), ("warm", "batch"), ("cache_hit", "single")):
                values.append(statistics.median(request["elapsed_ms"] for r in rows for request in r["requests"]
                                                if request["phase"] == phase and request["kind"] == kind))
            values += [statistics.median(r["memory_stages"]["after_cache_and_parity"][key] for r in rows) / 2**20
                       for key in ("peak_working_set_bytes", "peak_commit_bytes")]
            lines.append("| %s | %s | %.3f | %.2f | %.2f | %.2f | %.2f | %.1f | %.1f |" %
                         (transport, variant, *values))
    lines += ["", "Medians across %d fresh processes per row; warm calls pool %d samples per process. "
              "Startup includes Python, imports, preload, protocol readiness/handshake and the first answer. "
              "Peak counters cover the entire server lifetime through untimed parity and cache checks. "
              "Private commit excludes shared file mappings; working set includes resident mapped pages." %
              (report["process_runs"], report["repeats"]), "",
              "| transport | service | outside Router, warm single ms | outside Router, warm batch ms | "
              "outside Router, cache hit ms |", "|---|---|---:|---:|---:|"]
    for transport in report["transports"]:
        for variant in report["variants"]:
            rows = [r for r in report["runs"] if r["transport"] == transport and r["variant"] == variant]
            values = []
            for phase, kind in (("warm", "single"), ("warm", "batch"), ("cache_hit", "single")):
                values.append(statistics.median(request["outside_router_ms"] for r in rows for request in r["requests"]
                                                if request["phase"] == phase and request["kind"] == kind))
            lines.append("| %s | %s | %.3f | %.3f | %.3f |" % (transport, variant, *values))
    checks = report["parity_vs_first_torch"]
    lines += ["", "Outside-Router time is client round-trip minus an instrumented Router call: it includes "
              "request serialization, protocol/framework validation and response serialization, plus scheduling. "
              "It is not a network-only measurement. Router wrappers add two perf_counter reads and one list "
              "append; raw logit capture is enabled only on untimed parity calls.", "",
              "Shared typed-answer labels and schemas agree across transports/variants; the HTTP-only "
              "x_jev_confidence extension is excluded when comparing HTTP with MCP. Maximum absolute differences: "
              "public answer numbers %.9g, raw logits %.9g, unrounded probabilities %.9g, action probabilities %.9g. "
              "Declared absolute tolerances: public answers %.9g (four-decimal wire rounding), raw logits %.9g, "
              "unrounded probabilities/actions %.9g. Four hand-written states cover choice, score and yes/no; "
              "this is numerical parity, not an accuracy evaluation." % (
                  max(x["max_abs_public_answer_number"] for x in checks),
                  max(x["max_abs_logits"] for x in checks), max(x["max_abs_probabilities"] for x in checks),
                  max(x["max_abs_action_probabilities"] for x in checks),
                  PUBLIC_TOLERANCE, LOGIT_TOLERANCE, PROBABILITY_TOLERANCE), "",
              "The full-package and standalone ONNX services execute the same canonical native-tokenizer "
              "Router/HTTP/MCP code and graph. Their comparison controls runtime/backend changes. The frozen "
              "main service uses the default Torch backend. Mixed F16/F32 checkpoint weights are cast by the "
              "Torch CPU loader to fp32; the exported graph has FLOAT floating initializers and outputs. "
              "Loaded Torch dtype/no-AMP and ORT CPU provider/output types/thread count are asserted. "
              "Cache hit measurements enable the in-memory "
              "DecisionCache after cache-off tests; its counters assert one miss and the requested hits. "
              "The checkpoint remains resident during cache timing.", "",
              "Baseline commit: `%s`. Raw requests/responses, process order, source/artifact/wheel/harness hashes, "
              "Windows stage counters, versions, stderr and the command are in the adjacent JSON. "
              "The workstation was otherwise uncontrolled; small differences are timing noise." % report["baseline_revision"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--light-wheel", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--process-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cache-repeats", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--baseline-revision", default=BASE)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--transports", nargs="+", choices=("http", "mcp"), default=["http", "mcp"])
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--child-config", type=Path)
    args = parser.parse_args()
    if args.child_config:
        child(json.loads(args.child_config.read_text(encoding="utf-8")))
        return
    if args.self_check:
        assert numeric_difference({"a": [1, "label"]}, {"a": [1.25, "label"]}) == .25
        try:
            numeric_difference({"a": "label"}, {"a": "other"})
        except AssertionError:
            pass
        else:
            raise AssertionError("Changed decisions must fail")
        row = {"question": "q", "logits": [1.], "probabilities": [.5], "action_probabilities": [.5]}
        assert all(value == 0 for value in record_difference([row], [row]).values())
        assert process_memory(os.getpid())["peak_working_set_bytes"] > 0
        print("ONNX service benchmark self-check passed.")
        return
    if not all((args.model, args.onnx, args.light_wheel, args.out)):
        parser.error("--model, --onnx, --light-wheel and --out are required")
    if "baseline-torch" not in args.variants:
        parser.error("baseline-torch is required for parity")
    if min(args.threads, args.process_runs, args.repeats, args.cache_repeats) < 1 or args.warmups < 0:
        parser.error("Counts/threads must be positive; warmups must be nonnegative")
    args.model, args.onnx, args.light_wheel, args.out = [path.resolve() for path in
                                                      (args.model, args.onnx, args.light_wheel, args.out)]
    workspace = ROOT / ".venv" / "onnx-services-bench"
    workspace.mkdir(parents=True, exist_ok=True)
    baseline = baseline_tree(args.baseline_revision, workspace / "baseline")
    light = extract_wheel(args.light_wheel, workspace / ("wheel-" + sha256(args.light_wheel)[:12]))
    sources = {"baseline-torch": baseline, "full-onnx": ROOT, "lite-onnx": light}
    canonical_manifest = {}
    bundled = light / "laya_onnx" / "_runtime"
    for path in sorted(bundled.rglob("*.py")):
        relative = path.relative_to(bundled)
        if relative == Path("__init__.py"):
            continue  # Standalone init intentionally selects its ONNX default.
        canonical = ROOT / "laya" / relative
        actual, expected = sha256(path), sha256(canonical)
        assert actual == expected, "Stale wheel module: %s; rebuild before timing" % relative
        canonical_manifest[str(relative)] = {"canonical_sha256": expected, "wheel_sha256": actual}
    assert "onnx_agent.py" in canonical_manifest and "mcp\\server.py" in canonical_manifest
    artifacts = [args.model / "model.safetensors", args.model / "rl_agent_config.json"]
    artifacts += sorted((args.model / "tokenizer").glob("*.json"))
    artifacts += sorted(args.onnx.parent.glob(args.onnx.name + "*"))
    with (args.model / "model.safetensors").open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    report = {"command": subprocess.list2cmdline(sys.argv), "started_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_revision": args.baseline_revision, "threads": args.threads, "variants": args.variants,
              "transports": args.transports, "process_runs": args.process_runs, "repeats": args.repeats,
              "warmups": args.warmups, "cache_repeats": args.cache_repeats,
              "precision": "fp32 CPU; no quantization, no autocast", "cache_off_baseline": True,
              "checkpoint_dtypes": sorted({item["dtype"] for key, item in header.items() if key != "__metadata__"}),
              "artifacts": {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts},
              "wheel": {"path": str(args.light_wheel), "sha256": sha256(args.light_wheel)},
              "wheel_canonical_source_parity": canonical_manifest,
              "harness_sha256": sha256(Path(__file__)), "helper_sha256": sha256(Path(__file__).with_name("bench_loading_memory.py")),
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                              "cpu": platform.processor(), "logical_cpus": os.cpu_count()},
              "questions": QUESTIONS, "states": STATE_OBJECTS, "request_controls": CONTROLS,
              "tolerances": {"public_abs": PUBLIC_TOLERANCE, "logits_abs": LOGIT_TOLERANCE,
                             "probabilities_and_actions_abs": PROBABILITY_TOLERANCE}, "runs": []}
    # The source checkpoint contains mixed F16/F32 weights. The CPU loader casts
    # them to fp32; the child asserts the instantiated dtype and disabled AMP.
    assert set(report["checkpoint_dtypes"]) <= {"F16", "F32"}, report["checkpoint_dtypes"]
    import onnx
    graph = onnx.load(str(args.onnx), load_external_data=False)
    initializer_types = sorted({onnx.TensorProto.DataType.Name(value.data_type)
                                for value in graph.graph.initializer})
    report["graph_initializer_dtypes"] = initializer_types
    report["graph_output_dtypes"] = [onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
                                      for value in graph.graph.output]
    assert "FLOAT" in initializer_types
    assert set(initializer_types) <= {"FLOAT", "INT32", "INT64", "BOOL"}, initializer_types
    assert report["graph_output_dtypes"] == ["FLOAT", "FLOAT"], report["graph_output_dtypes"]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("LAYA_") or key in ("HF_TOKEN", "PYTHONPATH", "USE_TORCH"):
            env.pop(key)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0",
               LAYA_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               LAYA_DEVICE="cpu", LAYA_HOST="127.0.0.1", LAYA_LOG_LEVEL="warning", LAYA_PRELOAD="1",
               LAYA_MODELS="en", LAYA_DEFAULT_MODEL="en", LAYA_MAX_LOADED="1", LAYA_CACHE="0",
               LAYA_EXTRA_MODELS=json.dumps({"en": str(args.model)}), LAYA_ONNX_PATHS=json.dumps({"en": str(args.onnx)}))
    report["controlled_environment"] = {key: value for key, value in env.items() if key.startswith("LAYA_") or key in
                                        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
                                         "USE_TF", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.process_runs):
        order = list(args.variants) if repeat % 2 == 0 else list(reversed(args.variants))
        for transport in args.transports:
            for variant in order:
                row = run_service(args, variant, transport, repeat + 1, sources[variant], workspace, env)
                report["runs"].append(row)
                args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print("%s %s run %d: first answer %.3f s; peak WS %.1f MiB" %
                      (transport, variant, repeat + 1, row["startup_to_first_answer_s"],
                       row["memory_stages"]["after_cache_and_parity"]["peak_working_set_bytes"] / 2**20), flush=True)
    report["parity_vs_first_torch"] = parity(report["runs"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
