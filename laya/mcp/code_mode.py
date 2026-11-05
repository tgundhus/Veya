"""Optional local JavaScript composition over the existing validated MCP operations."""
import inspect
import json
import math
import multiprocessing
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
from dataclasses import asdict, dataclass

from ..errors import DecisionCacheError
from . import tools
from .tools import ToolError


_OPERATIONS = ("predict", "predict_batch", "route", "route_batch", "shortlist",
               "tournament", "predict_long", "preset", "decide", "status")
_INJECTED = frozenset(("router", "agent", "preset_builder", "embed_fn", "loaded",
                       "preload", "performance_mode", "backend", "hooks_timeout"))
_TERMINAL_ERRORS = frozenset(("decision_cache_unavailable", "busy", "upstream_error",
                              "timeout", "unreachable", "unauthorized", "internal_error", "models_not_ready",
                              "code_resource_limit"))


class CodeModeError(ToolError):
    """A bounded local-code execution failed; details never include host exceptions."""


@dataclass(frozen=True)
class CodeModeLimits:
    """Host-owned budgets. No script can raise these values."""
    wall_seconds: float = 120.0
    cpu_seconds: float = 2.0
    memory_bytes: int = 64 * 1024 * 1024
    stack_bytes: int = 512 * 1024
    max_script_bytes: int = 64 * 1024
    max_input_bytes: int = 1024 * 1024
    max_output_bytes: int = 1024 * 1024
    max_call_bytes: int = 1024 * 1024
    max_total_bytes: int = 8 * 1024 * 1024
    max_calls: int = 32
    max_pending_jobs: int = 10000
    max_states: int = 64
    max_questions: int = 64
    max_choice_options: int = 100
    max_tournament_options: int = 255
    max_score_levels: int = 32
    max_total_options: int = 512
    max_token_budget: int = 8192
    max_batch_tokens: int = 131072
    max_state_chars: int = 50000
    max_scan_windows: int = 64

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be a positive finite number" % name)
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value <= 0:
                raise ValueError("%s must be a positive finite number" % name)
            if name not in ("wall_seconds", "cpu_seconds") and not isinstance(value, int):
                raise ValueError("%s must be an integer" % name)
        if self.memory_bytes < 1024 * 1024 or self.stack_bytes < 64 * 1024:
            raise ValueError("memory_bytes needs at least 1 MiB and stack_bytes at least 64 KiB")


def _json_bytes(value, maximum, *, input_boundary=False):
    """Bound JSON structure before encoding and reject lossy JS numeric input."""
    count = 0
    approximate = 0

    def visit(item, depth):
        nonlocal count, approximate
        count += 1
        if depth > 32 or count > 50000:
            raise CodeModeError("code_input_limit", "JSON nesting or item count exceeds the code-mode limit")
        if item is None or isinstance(item, bool):
            approximate += 5
        elif isinstance(item, str):
            if len(item) > maximum:
                raise CodeModeError("code_input_limit", "JSON exceeds the code-mode byte limit")
            approximate += len(item.encode("utf-8"))
        elif isinstance(item, int):
            if abs(item) > 9007199254740991:
                raise CodeModeError("code_invalid_json", "JSON integers must fit the JavaScript safe-integer range")
            approximate += 20
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise CodeModeError("code_invalid_json", "JSON numbers must be finite")
            if input_boundary and item.is_integer():
                raise CodeModeError("code_invalid_json", "Use JSON integers for whole-number inputs; JavaScript cannot preserve 1.0 versus 1")
            approximate += 24
        elif isinstance(item, (list, dict)):
            approximate += 2 + len(item)
            if isinstance(item, dict):
                # ECMAScript enumerates array-index keys first, in ascending order.
                # Reject a lossy crossing rather than silently changing positional
                # questions, choice criteria or authoritative decision-cache keys.
                indices = [key for key in item if isinstance(key, str) and len(key) <= 10 and key.isascii()
                           and key.isdecimal() and str(int(key)) == key and int(key) < 4294967295]
                index_keys = set(indices)
                if list(item) != sorted(indices, key=int) + [key for key in item if key not in index_keys]:
                    raise CodeModeError("code_invalid_json", "Object key order must match JavaScript array-index ordering")
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise CodeModeError("code_invalid_json", "JSON object keys must be strings")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        else:
            raise CodeModeError("code_invalid_json", "Only JSON values may cross the code-mode boundary")
        if approximate > maximum:
            raise CodeModeError("code_input_limit", "JSON exceeds the code-mode byte limit")

    try:
        visit(value, 0)
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (UnicodeError, RecursionError, ValueError, OverflowError) as exc:
        raise CodeModeError("code_invalid_json", "Only finite Unicode JSON values are supported") from exc
    if len(raw) > maximum:
        raise CodeModeError("code_input_limit", "JSON exceeds the code-mode byte limit")
    return raw


def _parse(raw):
    def invalid(_):
        raise ValueError("nonfinite number")
    try:
        return json.loads(raw, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CodeModeError("code_invalid_json", "The sandbox returned invalid JSON") from exc


def _operation(name):
    return getattr(tools, "laya_" + name)


def _parameters(name):
    return {key: value for key, value in inspect.signature(_operation(name)).parameters.items()
            if key not in _INJECTED}


def code_mode_description():
    """Generate argument names and requiredness from the authoritative tool functions."""
    declarations = []
    types = {"state": "JsonObject", "questions": "{[id:string]:Question}", "schema": "JsonObject",
             "requests": "DecisionRequest[]", "model": "string|null", "task": "string|null",
             "lang": "string|null", "lang_guess": "string|null", "preset": "string",
             "min_confidence": "number|null", "max_len": "number|null", "head_max_len": "number|null",
             "window": "number|null", "stride": "number|null", "group_size": "number", "k": "number|null",
             "batch_size": "number|null", "sort_by_length": "boolean", "explain": "boolean"}
    results = {"predict": "Prediction", "predict_batch": "BatchPrediction", "route": "Route",
               "route_batch": "{decisions:Route[];model_counts:{[model:string]:number}}"}
    for name in _OPERATIONS:
        parameters = []
        for key, parameter in _parameters(name).items():
            optional = "?" if parameter.default is not inspect.Parameter.empty else ""
            parameters.append("%s%s: %s" % (key, optional, types.get(key, "Json")))
        declarations.append("  %s(args%s: {%s}): Promise<%s>;" % (
            name, "?" if not parameters else "", "; ".join(parameters), results.get(name, "Json")))
    return ("Execute one bounded, fully local JavaScript decision workflow. Submit an async function "
            "async (input, veya) => { ...; return jsonValue; }. input is the separate JSON input argument. "
            "veya calls take one named-argument object and use the existing validated Veya operations. "
            "Use predict_batch for independent requests; host model calls are serialized. Await every call. "
            "The return value must be finite JSON. No filesystem, network, imports, environment or shell "
            "bindings exist. Required-cache and transport failures abort the entire workflow. "
            "Predictions are typed choice/score/noul decisions, not generated text. "
            "decide.schema supports object properties with enum, boolean, or bounded integer types. "
            "type Json = null | boolean | number | string | Json[] | {[key:string]:Json};\n"
            "type JsonObject = {[key:string]:Json};\n"
            "type Question = {type:'choice';instructions:string;criteria:{[label:string]:string|null};option_order?:number[]}"
            " | {type:'score';instructions:string;criteria:string[]}"
            " | {type:'noul';instructions:string;criteria?:{false?:string|null;true?:string|null};labels?:{false:string;true:string}};\n"
            "type DecisionRequest = {state:JsonObject;questions:{[id:string]:Question};model?:string;"
            "task?:string;lang?:string;lang_guess?:string;max_len?:number;head_max_len?:number};\n"
            "type Answer = {type:string;noul?:number;choice?:string;score?:number;confidence?:number;"
            "answer_confidence?:number;probabilities?:{[label:string]:number};low_confidence?:boolean};\n"
            "type Route = {model:string|null;repo?:string|null;reason?:string};\n"
            "type Prediction = {answers:{[id:string]:Answer};routing:Route;device?:string;latency_ms?:number};\n"
            "type BatchPrediction = {requests:Prediction[];model_counts:{[model:string]:number};total_latency_ms:number};\n"
            "declare const veya: {\n" + "\n".join(declarations) + "\n};\n"
            "Example: async input => { const r = await veya.predict_batch({requests:input.map(state => "
            "({state,questions:{urgent:{type:'noul',instructions:'Does this need attention today?'}}}))}); "
            "return r.requests.map(x => x.answers.urgent.noul); }")


def _preset_builder(name):
    from importlib import import_module
    namespace = import_module("..", __package__)
    return getattr(namespace, name)()


def _controls(item, router, min_confidence=None):
    selected = tools._routed_control(item.get("model"), tools.validate_model, router)
    result = {} if selected == "auto" else {"model": selected}
    for name, validator in (("task", tools.validate_task), ("lang", tools.validate_lang),
                            ("lang_guess", tools.validate_lang_guess),
                            ("min_confidence", tools.validate_min_confidence)):
        value = (min_confidence if name == "min_confidence" and min_confidence is not None
                 else item.get(name))
        value = tools._routed_control(value, validator, router) if name == "task" else validator(value)
        if value is not None:
            result[name] = value
    if selected != "auto":
        for name in ("task", "lang_guess"):
            if name in result:
                raise ToolError("invalid_" + name, name + " routes between checkpoints and cannot accompany a pinned model")
    for name in ("max_len", "head_max_len"):
        value = tools.validate_budget(item.get(name), name)
        if value is not None:
            result[name] = value
    return result


class _PreparedRouter:
    """Use already-authorized hits once, and forward only the preflighted miss inputs."""
    def __init__(self, router, prepared, maximum):
        self.router, self.prepared, self.maximum = router, prepared, maximum

    def __getattr__(self, name):
        return getattr(self.router, name)

    def _identity(self, state, questions, kwargs):
        return _json_bytes([state, questions, _controls(kwargs, self.router)], self.maximum)

    def predict(self, state, questions, **kwargs):
        identity = self._identity(state, questions, kwargs)
        if identity not in self.prepared:
            raise CodeModeError("code_resource_limit", "Operation inputs changed after resource preflight")
        hit = self.prepared[identity]
        return hit if hit is not None else self.router.predict(state, questions, **kwargs)

    def predict_batch(self, requests, **kwargs):
        result = []
        missing, positions = [], []
        for index, item in enumerate(requests):
            controls = _controls(item, self.router, kwargs.get("min_confidence"))
            identity = self._identity(item["state"], item["questions"], controls)
            if identity not in self.prepared:
                raise CodeModeError("code_resource_limit", "Operation inputs changed after resource preflight")
            hit = self.prepared[identity]
            result.append(hit)
            if hit is None:
                missing.append(item)
                positions.append(index)
        if missing:
            values = self.router.predict_batch(missing, **kwargs)
            if len(values) != len(missing):
                raise CodeModeError("code_operation_failed", "The Veya batch operation returned incomplete results")
            for index, value in zip(positions, values):
                result[index] = value
        return result


class CodeModeRunner:
    """One admitted workflow, one bounded host worker and a fresh guest process per call.

    A timed-out host prediction keeps its admission permit until it actually finishes.
    The caller fails at its deadline; model execution is never forcibly interrupted.
    """

    def __init__(self, limits=None):
        self.limits = limits if limits is not None else CodeModeLimits()
        if not isinstance(self.limits, CodeModeLimits):
            raise TypeError("limits must be CodeModeLimits")
        self._permit = threading.BoundedSemaphore(1)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veya-code-host")
        self._closed = False
        self._lifecycle = threading.Condition()
        self._active_workflows = 0
        self._caller_threads = {}
        self._host_thread = None

    def close(self):
        """Stop guest workflows and drain host predictions before their Router/cache closes."""
        identity = threading.get_ident()
        with self._lifecycle:
            if identity in self._caller_threads or identity == self._host_thread:
                raise RuntimeError("close must run outside this runner's active caller or host-operation thread")
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lifecycle:
            while self._active_workflows:
                self._lifecycle.wait()

    def _ensure_open(self):
        if self._closed:
            raise CodeModeError("code_unavailable", "Code mode is closing or closed")

    def _call_host(self, method, arguments, router):
        with self._lifecycle:
            self._host_thread = threading.get_ident()
        try:
            self._ensure_open()
            return self._invoke(method, arguments, router)
        finally:
            with self._lifecycle:
                self._host_thread = None
                self._lifecycle.notify_all()

    def _invoke(self, method, arguments, router):
        if method not in _OPERATIONS or not isinstance(arguments, dict):
            raise CodeModeError("code_invalid_call", "Unknown operation or non-object arguments")
        parameters = _parameters(method)
        if set(arguments) - set(parameters):
            raise CodeModeError("code_invalid_call", "Operation arguments include an unknown or host-only field")
        if method == "predict_batch" and not isinstance(arguments.get("sort_by_length", False), bool):
            raise ToolError("invalid_sort_by_length", "sort_by_length must be a boolean")
        for name, parameter in parameters.items():
            if parameter.default is inspect.Parameter.empty and name not in arguments:
                raise ToolError("invalid_request", "Missing operation argument: " + name)
        from ._code_budget import check_operation_budget
        from .device import router_agent
        from .remote import RemoteRouter
        from ..router import Router
        from ..consistency import DecisionCache
        from ..hooks import compose_hooks

        if type(router) not in (Router, RemoteRouter):
            raise CodeModeError("code_resource_limit", "Code mode requires the built-in local or remote Router")
        if callable(router.lang_guess) or any(type(hook) is not DecisionCache
                for hook in compose_hooks(getattr(router, "hooks", ()) or ())):
            raise CodeModeError("code_resource_limit", "Code mode requires cache-only hooks and declarative routing")
        items = check_operation_budget(method, arguments, router, self.limits, preflight=True)
        replayed, prepared = set(), {}
        replay_methods = ("predict", "predict_batch", "decide", "preset")
        if type(router) is Router and method not in ("route", "route_batch", "status"):
            for index, item in enumerate(items):
                controls = _controls(item, router, arguments.get("min_confidence") if method == "predict_batch" else None)
                identity = (_json_bytes([item["state"], item["questions"], controls], self.limits.max_call_bytes)
                            if method in replay_methods else None)
                hit = None
                if method in replay_methods:
                    hit = router.try_replay(item["state"], item["questions"], **controls)
                if hit is not None:
                    replayed.add(index)
                else:
                    route = router._route(item["state"], item["questions"], **{
                        key: value for key, value in controls.items() if key in ("model", "task", "lang", "lang_guess")})
                    agent = router_agent(router, route["model"])
                    if agent is None:
                        agent = router.load(route["model"])
                    namespace = __package__.rsplit(".", 1)[0]
                    onnx = sys.modules.get(namespace + ".onnx_agent")
                    builtin = (getattr(sys.modules.get(namespace + ".agent"), "Agent", None),
                               getattr(onnx, "ONNXAgent", None), getattr(onnx, "_NativeONNXAgent", None),
                               (getattr(sys.modules.get("laya_onnx"), "ONNXAgent", None)
                                if namespace == "laya_onnx._runtime" else None))
                    if type(agent) not in builtin or any(type(hook) is not DecisionCache
                            for hook in compose_hooks(getattr(agent, "hooks", ()) or ())):
                        raise CodeModeError("code_resource_limit", "Code mode requires a built-in checkpoint with cache-only hooks")
                if method in replay_methods:
                    prepared[identity] = hit
        check_operation_budget(method, arguments, router, self.limits, replayed=replayed)
        operation_router = (_PreparedRouter(router, prepared, self.limits.max_call_bytes)
                            if prepared else router)
        kwargs = dict(arguments, router=operation_router)
        if method == "preset":
            kwargs["preset_builder"] = _preset_builder
        return _operation(method)(**kwargs)

    def execute(self, code, input=None, *, router=None):
        """Execute a JavaScript function over JSON with only validated Veya capabilities."""
        if self._closed:
            raise CodeModeError("code_unavailable", "Code mode is closed")
        if not isinstance(code, str) or not code.strip():
            raise CodeModeError("code_invalid_script", "code must be a nonempty JavaScript function string")
        if len(code) > self.limits.max_script_bytes:
            raise CodeModeError("code_input_limit", "The script exceeds the code-mode byte limit")
        try:
            source_size = len(code.encode("utf-8"))
        except UnicodeError as exc:
            raise CodeModeError("code_invalid_script", "code must contain valid Unicode") from exc
        if source_size > self.limits.max_script_bytes:
            raise CodeModeError("code_input_limit", "The script exceeds the code-mode byte limit")
        encoded_input = _json_bytes(input, self.limits.max_input_bytes, input_boundary=True)
        try:
            import quickjs  # noqa: F401 -- fail before spawning when the extra is absent
            import psutil
        except ImportError as exc:
            raise CodeModeError("code_unavailable", "Install the local veya[codemode] extra to enable code mode") from exc
        release = True
        parent = child = process = None
        active_host = None
        caller = threading.get_ident()
        with self._lifecycle:
            self._ensure_open()
            if not self._permit.acquire(blocking=False):
                raise CodeModeError("code_busy", "A code workflow or timed-out host prediction is still running")
            self._active_workflows += 1
            self._caller_threads[caller] = self._caller_threads.get(caller, 0) + 1
        try:
            deadline = time.monotonic() + self.limits.wall_seconds
            calls = 0
            total_bytes = source_size + len(encoded_input)
            parent, child = multiprocessing.get_context("spawn").Pipe()
            from ._code_worker import run_worker
            worker_limits = asdict(self.limits)
            # Envelopes and JSON string escaping can expand an already-bounded payload.
            worker_limits["max_message_bytes"] = 6 * max(self.limits.max_call_bytes, self.limits.max_output_bytes) + 4096
            process = multiprocessing.get_context("spawn").Process(
                target=run_worker, args=(child, code, encoded_input.decode("utf-8"), _OPERATIONS, worker_limits),
                name="veya-code-sandbox", daemon=True)
            process.start()
            child.close()
            try:
                monitored = psutil.Process(process.pid)
            except psutil.NoSuchProcess:
                if process.is_alive():
                    raise CodeModeError("code_sandbox_failed", "The guest CPU monitor is unavailable") from None
                monitored = None
            cpu_started = None

            def check_cpu():
                if cpu_started is not None and monitored is not None:
                    try:
                        counters = monitored.cpu_times()
                    except psutil.NoSuchProcess:
                        if process.is_alive():
                            raise CodeModeError("code_sandbox_failed", "The guest CPU monitor is unavailable") from None
                        return
                    if counters.user + counters.system - cpu_started > self.limits.cpu_seconds:
                        raise CodeModeError("code_cpu_limit", "JavaScript exceeded its cumulative CPU limit")

            while True:
                self._ensure_open()
                remaining = deadline - time.monotonic()
                check_cpu()
                if remaining <= 0:
                    raise CodeModeError("code_timeout", "The code workflow exceeded its wall-time limit")
                if not parent.poll(min(remaining, .01)):
                    continue
                wire = parent.recv_bytes(worker_limits["max_message_bytes"])
                total_bytes += len(wire)
                if total_bytes > self.limits.max_total_bytes:
                    raise CodeModeError("code_output_limit", "The workflow exceeded its total byte limit")
                event = _parse(wire)
                if not isinstance(event, dict):
                    raise CodeModeError("code_sandbox_failed", "The sandbox returned an invalid event")
                if event.get("type") == "ready":
                    baseline = event.get("cpu_started")
                    if (cpu_started is not None or type(baseline) not in (int, float)
                            or not math.isfinite(baseline) or baseline < 0
                            or type(event.get("pid")) is not int or event["pid"] != process.pid):
                        raise CodeModeError("code_sandbox_failed", "The sandbox returned an invalid startup event")
                    if monitored is None:
                        cpu_started = baseline
                    else:
                        try:
                            observed = monitored.cpu_times()
                        except psutil.NoSuchProcess:
                            if process.is_alive():
                                raise CodeModeError("code_sandbox_failed", "The guest CPU monitor is unavailable") from None
                            # A short pure program may have queued its final result.
                            cpu_started = baseline
                        else:
                            cpu_started = min(baseline, observed.user + observed.system)
                    continue
                if cpu_started is None:
                    raise CodeModeError("code_sandbox_failed", "The sandbox did not initialize its CPU monitor")
                check_cpu()
                if event.get("type") == "result":
                    value = event.get("value")
                    if not isinstance(value, str):
                        raise CodeModeError("code_sandbox_failed", "The sandbox returned an invalid result")
                    result = _parse(value)
                    raw = _json_bytes(result, self.limits.max_output_bytes)
                    if time.monotonic() >= deadline:
                        raise CodeModeError("code_timeout", "The code workflow exceeded its wall-time limit")
                    self._ensure_open()
                    return result
                if event.get("type") != "call":
                    raise CodeModeError("code_script_failed", "JavaScript failed, exhausted a budget or left unresolved work")
                request = _parse(event.get("value", ""))
                raw = _json_bytes(request, self.limits.max_call_bytes)
                calls += 1
                if (calls > self.limits.max_calls or total_bytes > self.limits.max_total_bytes
                        or not isinstance(request, dict) or type(request.get("id")) is not int
                        or request.get("id") != calls):
                    raise CodeModeError("code_call_limit", "The workflow exceeded its call/byte limit or returned an invalid call")
                with self._lifecycle:
                    self._ensure_open()
                    if not process.is_alive():
                        raise CodeModeError("code_sandbox_failed", "The sandbox stopped before its host operation")
                    future = self._pool.submit(self._call_host, request.get("method"), request.get("args"), router)
                    active_host = future
                try:
                    while not future.done():
                        self._ensure_open()
                        check_cpu()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise CodeModeError("code_timeout", "The code workflow exceeded its wall-time limit")
                        wait_futures((future,), timeout=min(remaining, .01))
                    if time.monotonic() >= deadline:
                        raise CodeModeError("code_timeout", "The code workflow exceeded its wall-time limit")
                    self._ensure_open()
                    value = future.result()
                    response = {"id": calls, "result": value}
                except CodeModeError:
                    # Retain admission until actual work finishes. Never enqueue abandoned
                    # jobs on retries, or close a cache while its prediction still uses it.
                    if not future.done():
                        future.add_done_callback(lambda _: self._permit.release())
                        release = False
                    raise
                except DecisionCacheError as exc:
                    raise CodeModeError("decision_cache_unavailable", "required decision storage is unavailable") from exc
                except ToolError as exc:
                    if exc.code in _TERMINAL_ERRORS or isinstance(exc, CodeModeError):
                        raise CodeModeError(exc.code, exc.message) from exc
                    response = {"id": calls, "error": {"code": exc.code, "message": exc.message}}
                except Exception as exc:
                    raise CodeModeError("code_operation_failed", "The Veya operation failed") from exc
                raw = _json_bytes(response, self.limits.max_output_bytes)
                total_bytes += len(raw)
                if total_bytes > self.limits.max_total_bytes:
                    raise CodeModeError("code_output_limit", "The workflow exceeded its total byte limit")
                parent.send_bytes(raw)
        except CodeModeError:
            raise
        except Exception as exc:
            raise CodeModeError("code_sandbox_failed", "The local sandbox stopped unexpectedly") from exc
        finally:
            if release and active_host is not None and not active_host.done():
                active_host.add_done_callback(lambda _: self._permit.release())
                release = False
            try:
                if parent is not None:
                    parent.close()
                if child is not None:
                    child.close()
                if process is not None and process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                    process.close()
            finally:
                if release:
                    self._permit.release()
                with self._lifecycle:
                    self._active_workflows -= 1
                    self._caller_threads[caller] -= 1
                    if not self._caller_threads[caller]:
                        del self._caller_threads[caller]
                    self._lifecycle.notify_all()
