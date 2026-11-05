"""The optional local JavaScript boundary composes validated decisions and keeps failures fatal."""
import inspect
import asyncio
import multiprocessing
import threading
import time
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("quickjs")

from laya import DecisionCache, Router
from laya.errors import DecisionCacheError
from laya.mcp import tools
from laya.mcp.code_mode import CodeModeError, CodeModeLimits, CodeModeRunner, code_mode_description
from laya.mcp.remote import RemoteError
from laya.onnx_agent import ONNXAgent


QUESTIONS = {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}


class Predictor:
    cfg = {"max_len": 512, "head_max_len": 192}
    hooks = ()

    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kwargs):
        self.calls += 1
        return {"answers": {key: {"type": "noul", "noul": 0.51} for key in questions}, "usage": {}}

    def predict_batch(self, states, questions, **kwargs):
        return [self.system_one(state, questions, **kwargs) for state in states]


@pytest.fixture
def runtime():
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=8, cpu_seconds=.3))
    predictor = object.__new__(ONNXAgent)
    predictor.cfg, predictor.hooks, predictor.calls = Predictor.cfg, (), 0
    predictor.system_one = MethodType(Predictor.system_one, predictor)
    predictor.predict_batch = MethodType(Predictor.predict_batch, predictor)
    cache = DecisionCache(required=True, maxsize=None, fingerprint="code-policy-v1")
    router = Router(hooks=[cache])
    router.attach("english", predictor)
    try:
        yield runner, router, predictor
    finally:
        runner.close()
        cache.close()


def request(text="hello"):
    return {"state": {"text": text}, "questions": QUESTIONS, "model": "english"}


def test_batched_workflow_filters_and_replays_authoritative_decisions(runtime):
    runner, router, predictor = runtime
    code = """async (input, v) => {
      const batch = await v.predict_batch({requests: input});
      return batch.requests.map(r => r.answers.urgent.noul).filter(p => p >= .5);
    }"""
    rows = [request("one"), request("two")]
    assert runner.execute(code, rows, router=router) == [.51, .51]
    assert runner.execute(code, rows, router=router) == [.51, .51]
    assert predictor.calls == 2


def test_dependent_calls_and_global_api_use_same_validated_tools(runtime):
    runner, router, _ = runtime
    result = runner.execute("""async input => {
      const route = await veya.route({state: input.state, questions: input.questions});
      const prediction = await veya.predict({...input, model: route.model});
      return prediction.answers.urgent.noul;
    }""", request(), router=router)
    assert result == .51


def test_catchable_validation_does_not_reach_inference(runtime):
    runner, router, predictor = runtime
    result = runner.execute("""async () => {
      try { await veya.predict({state: null, questions: {}}); }
      catch (error) { return error.code; }
      return 'unexpected';
    }""", router=router)
    assert result == "invalid_state"
    assert predictor.calls == 0


@pytest.mark.parametrize("error", [DecisionCacheError("private-path"),
    tools.ToolError("decision_cache_unavailable", "required decision storage is unavailable"),
    RemoteError("busy", "required decision storage is unavailable"), RemoteError("upstream_error", "remote failed")])
def test_required_or_uncertain_transport_failure_cannot_be_caught_into_replacement(runtime, monkeypatch, error):
    runner, router, _ = runtime
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(router, "predict", fail)
    with pytest.raises(CodeModeError) as exc:
        runner.execute("""async input => {
          try { return await veya.predict(input); }
          catch (_) { await veya.predict(input); return {replacement: true}; }
        }""", request(), router=router)
    assert len(attempts) == 1
    assert "private-path" not in str(exc.value)


@pytest.mark.parametrize("name", ["router", "agent", "preset_builder", "embed_fn", "hooks", "__proto__"])
def test_host_capabilities_cannot_be_supplied_as_arguments(runtime, name):
    runner, router, predictor = runtime
    with pytest.raises(CodeModeError, match="host-only"):
        runner.execute("async input => await veya.predict(input)", {**request(), name: {}}, router=router)
    assert predictor.calls == 0


def test_no_ambient_host_capabilities_and_api_is_frozen(runtime):
    runner, router, _ = runtime
    assert runner.execute("""async () => {
      return {process:typeof process, require:typeof require, fetch:typeof fetch,
              std:typeof std, os:typeof os, Python:typeof Python, WebAssembly:typeof WebAssembly,
              frozen:Object.isFrozen(veya), prototype:Object.getPrototypeOf(veya)};
    }""", router=router) == {"process": "undefined", "require": "undefined", "fetch": "undefined",
        "std": "undefined", "os": "undefined", "Python": "undefined", "WebAssembly": "undefined",
        "frozen": True, "prototype": None}


@pytest.mark.parametrize("code", [
    "async () => { while(true) {} }",
    "async () => { while(true) await Promise.resolve(); }",
    "async () => ({get secret() { while(true) {} }})",
    "async () => ({toJSON() { while(true) {} }})",
    "async () => { function f(){return f()} return f(); }",
    "async () => { const rows=[]; while(true) rows.push(new Array(100000).fill('payload')); }",
])
def test_guest_cpu_memory_stack_and_serialization_are_bounded(code):
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=3, cpu_seconds=.05, memory_bytes=8 * 1024 * 1024))
    begun = time.monotonic()
    try:
        with pytest.raises(CodeModeError):
            runner.execute(code)
    finally:
        runner.close()
    assert time.monotonic() - begun < 5


def test_job_budget_is_cumulative_even_after_host_calls(runtime):
    _, router, _ = runtime
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=5, cpu_seconds=1, max_pending_jobs=15))
    try:
        with pytest.raises(CodeModeError):
            runner.execute("""async input => {
              for (let i=0;i<20;i++) { await veya.predict(input); await Promise.resolve(); }
              return true;
            }""", request(), router=router)
    finally:
        runner.close()


def test_call_quota_is_parent_owned_and_not_catchable(runtime):
    _, router, predictor = runtime
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=5, max_calls=2))
    try:
        with pytest.raises(CodeModeError) as exc:
            runner.execute("""async input => {
              try { for (let i=0;i<10;i++) await veya.predict({...input,state:{text:String(i)}}); }
              catch (_) { return 'quota hidden'; }
              return true;
            }""", request(), router=router)
        assert exc.value.code == "code_call_limit"
        assert predictor.calls == 2
    finally:
        runner.close()


@pytest.mark.parametrize("code", ["async () => new Promise(() => {})", "async input => { veya.predict(input); return true; }"])
def test_unresolved_or_fire_and_forget_work_is_refused(runtime, code):
    runner, router, predictor = runtime
    with pytest.raises(CodeModeError):
        runner.execute(code, request(), router=router)
    assert predictor.calls == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2 ** 53, 1.0, -0.0, "\ud800", {1: "value"}])
def test_lossy_or_non_json_inputs_are_refused(runtime, value):
    runner, _, _ = runtime
    with pytest.raises(CodeModeError):
        runner.execute("async input => input", value)


def test_numeric_key_reordering_is_refused_before_decisions(runtime):
    runner, router, predictor = runtime
    ordered = {"pick": {"type": "choice", "instructions": "Choose", "criteria": {"10": "first", "2": "second"}}}
    with pytest.raises(CodeModeError, match="key order"):
        runner.execute("async input => await veya.predict(input)", {**request(), "questions": ordered}, router=router)
    assert predictor.calls == 0


def test_depth_script_and_output_limits():
    runner = CodeModeRunner(CodeModeLimits(max_script_bytes=100, max_output_bytes=100))
    try:
        with pytest.raises(CodeModeError):
            runner.execute(" " * 101)
        with pytest.raises(CodeModeError):
            runner.execute("async () => 'x'.repeat(200)")
        nested = {}
        for _ in range(35):
            nested = {"nested": nested}
        with pytest.raises(CodeModeError):
            runner.execute("async input => input", nested)
    finally:
        runner.close()


def test_request_arguments_are_snapshotted_before_mutation(runtime):
    runner, router, _ = runtime
    assert runner.execute("""async input => {
      const p = veya.predict(input); input.questions = {};
      return (await p).answers.urgent.noul;
    }""", request(), router=router) == .51


def test_fresh_contexts_do_not_leak_globals_between_calls(runtime):
    runner, _, _ = runtime
    assert runner.execute("async () => { globalThis.secret='previous'; return true; }") is True
    assert runner.execute("async () => typeof secret") == "undefined"


def test_host_timeout_returns_and_retains_admission_until_actual_completion(runtime, monkeypatch):
    _, router, predictor = runtime
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=.5))
    entered, release = threading.Event(), threading.Event()
    original = router.predict

    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(router, "predict", slow)
    begun = time.monotonic()
    try:
        with pytest.raises(CodeModeError) as failure:
            runner.execute("async input => await veya.predict(input)", request(), router=router)
        assert entered.is_set()
        assert failure.value.code == "code_timeout"
        assert time.monotonic() - begun < 2
        with pytest.raises(CodeModeError) as busy:
            runner.execute("async () => true")
        assert busy.value.code == "code_busy"
        assert predictor.calls == 0
    finally:
        release.set()
        runner.close()
    assert predictor.calls == 1


def test_close_and_worker_cleanup(runtime):
    runner, _, _ = runtime
    before = {p.pid for p in multiprocessing.active_children()}
    assert runner.execute("async () => 42") == 42
    assert {p.pid for p in multiprocessing.active_children()} == before
    runner.close()
    with pytest.raises(CodeModeError):
        runner.execute("async () => 42")


def test_tool_declarations_follow_existing_signatures():
    description = code_mode_description()
    assert "router:" not in description
    assert "preset_builder:" not in description
    for name in ("predict", "predict_batch", "route", "route_batch", "shortlist", "tournament", "predict_long", "preset", "decide", "status"):
        assert name + "(args" in description
        for key in inspect.signature(getattr(tools, "laya_" + name)).parameters:
            if key not in ("router", "agent", "preset_builder", "embed_fn", "loaded", "preload", "performance_mode", "backend", "hooks_timeout"):
                assert key + ":" in description or key + "?:" in description


@pytest.mark.parametrize("kwargs", [{"wall_seconds": True}, {"cpu_seconds": float("inf")},
    {"max_calls": 1.1}, {"max_calls": 0}, {"memory_bytes": 64}, {"stack_bytes": 16}])
def test_invalid_host_limits_fail_before_running(kwargs):
    with pytest.raises(ValueError):
        CodeModeLimits(**kwargs)


@pytest.mark.parametrize("code", [
    "(()=>{throw {toString(){while(true){}}};})()",
    "async()=>{queueMicrotask(()=>{throw {toString(){while(true){}}}});await Promise.resolve();return 42;}",
    "async()=>{Error.prototype.toString=()=>{while(true){}}; throw new Error('hidden');}",
    "async()=>{Object.defineProperty(Error.prototype,'stack',{get(){while(true){}}}); while(true){}}",
])
def test_native_exception_conversion_cannot_escape_cpu_budget(code):
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=2, cpu_seconds=.03))
    started = time.monotonic()
    try:
        with pytest.raises(CodeModeError) as error:
            runner.execute(code)
        assert error.value.code != "code_timeout"
    finally:
        runner.close()
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("allocation", ["Pipe", "Process"])
def test_failed_process_allocation_releases_admission(runtime, monkeypatch, allocation):
    runner, _, _ = runtime
    context = multiprocessing.get_context("spawn")
    original = getattr(context, allocation)
    monkeypatch.setattr(context, allocation, lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private-path")))
    with pytest.raises(CodeModeError) as error:
        runner.execute("async()=>42")
    assert "private-path" not in str(error.value)
    monkeypatch.setattr(context, allocation, original)
    assert runner.execute("async()=>42") == 42


def test_parent_interruption_keeps_host_admission_until_work_finishes(runtime, monkeypatch):
    runner, router, _ = runtime
    entered, release = threading.Event(), threading.Event()
    original_wait = __import__("laya.mcp.code_mode", fromlist=["wait_futures"]).wait_futures

    def slow(*args, **kwargs):
        entered.set()
        release.wait(4)
        return {}

    def interrupted(*args, **kwargs):
        if entered.wait(.1):
            raise KeyboardInterrupt()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(runner, "_invoke", slow)
    monkeypatch.setattr("laya.mcp.code_mode.wait_futures", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.execute("async()=>await veya.status({})", router=router)
        with pytest.raises(CodeModeError) as busy:
            runner.execute("async()=>42")
        assert busy.value.code == "code_busy"
    finally:
        release.set()
        runner.close()


def test_single_tool_mcp_mode_calls_existing_validated_operations(runtime, monkeypatch):
    pytest.importorskip("mcp")
    from laya.mcp import server
    runner, router, _ = runtime
    monkeypatch.setattr(server, "_ROUTER", router)
    application = server.create_code_server(runner)

    async def exercise():
        catalog = await application.list_tools()
        assert [tool.name for tool in catalog] == ["veya_execute"]
        assert "code" in catalog[0].input_schema["required"]
        response = await application.call_tool("veya_execute", {
            "code": "async input => (await veya.predict(input)).answers.urgent.noul", "input": request()})
        assert any("0.51" in block.text for block in response.content)
        assert len(await server.server.list_tools()) == 10

    asyncio.run(exercise())


def test_embedded_server_owns_only_its_own_runner_and_supports_lifespan_restart(monkeypatch):
    pytest.importorskip("mcp")
    from laya.mcp import server

    class Spy:
        _closed = False
        def close(self):
            self._closed = True
        def execute(self, *args, **kwargs):
            return True

    created = []
    def construct():
        runner = Spy()
        created.append(runner)
        return runner

    monkeypatch.setattr("laya.mcp.code_mode.CodeModeRunner", construct)
    owned = server.create_code_server()
    external = Spy()
    borrowed = server.create_code_server(external)

    async def exercise():
        async with owned.settings.lifespan(owned):
            assert not created[-1]._closed
        assert created[-1]._closed
        async with owned.settings.lifespan(owned):
            assert len(created) == 2
            assert not created[-1]._closed
        assert all(runner._closed for runner in created)
        async with borrowed.settings.lifespan(borrowed):
            pass
        assert not external._closed

    asyncio.run(exercise())


@pytest.mark.parametrize("args,mode", [(["--code-max-calls", "1"], "classic"),
    (["--mode", "code", "--code-cpu-seconds", "nan"], "classic"), ([], "typo")])
def test_bad_code_cli_configuration_refuses_startup_before_model_load(monkeypatch, args, mode):
    pytest.importorskip("mcp")
    from laya.mcp import server
    monkeypatch.setenv("LAYA_MCP_MODE", mode)
    monkeypatch.setattr(server, "_ensure_router", lambda: pytest.fail("configuration must fail before loading"))
    with pytest.raises(SystemExit) as error:
        server.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("method,arguments,code", [
    ("predict", {**request(), "task": "typed_decisions"}, "invalid_task"),
    ("predict", {**request(), "lang_guess": "fr"}, "invalid_lang_guess"),
    ("predict", {**request(), "min_confidence": -1}, "invalid_min_confidence"),
    ("predict_batch", {"requests": [request()], "batch_size": 0}, "invalid_batch_size"),
    ("predict_batch", {"requests": [request()], "min_confidence": -1}, "invalid_min_confidence"),
    ("predict_batch", {"requests": [request()], "sort_by_length": "yes"}, "invalid_sort_by_length"),
])
def test_invalid_operation_controls_fail_before_any_cold_model_load(monkeypatch, method, arguments, code):
    router = Router()
    loaded = []
    def forbidden(*args, **kwargs):
        loaded.append(args)
        raise AssertionError("Invalid input must not load a checkpoint")
    monkeypatch.setattr(router, "load", forbidden)
    runner = CodeModeRunner()
    try:
        result = runner.execute("""async input => {
          try { return await veya[input.method](input.arguments); }
          catch(error) { return error.code; }
        }""", {"method": method, "arguments": arguments}, router=router)
        assert result == code
        assert not loaded
    finally:
        runner.close()


def test_large_host_token_budget_is_rejected_before_loading_or_forwarding(monkeypatch):
    router = Router()
    monkeypatch.setattr(router, "load", lambda *args: pytest.fail("oversized guest budget reached a load"))
    runner = CodeModeRunner()
    try:
        with pytest.raises(CodeModeError) as error:
            runner.execute("async input => await veya.predict(input)", {**request(), "max_len": 10 ** 15}, router=router)
        assert error.value.code == "code_resource_limit"
    finally:
        runner.close()


def test_mutated_object_json_hook_cannot_forge_host_protocol_completion(runtime):
    runner, router, predictor = runtime
    result = runner.execute("""async input => {
      const pending = veya.predict(input);
      Object.prototype.toJSON = () => ({type:'result', value:'42'});
      return await pending;
    }""", request(), router=router)
    assert result == {"type": "result", "value": "42"}
    assert predictor.calls == 1


def test_mutating_map_methods_cannot_replace_private_rpc_tracking(runtime):
    runner, router, predictor = runtime
    assert runner.execute("""async input => {
      Map.prototype.get = () => undefined;
      Map.prototype.set = () => {throw new Error('poisoned')};
      return (await veya.predict(input)).answers.urgent.noul;
    }""", request(), router=router) == .51
    assert predictor.calls == 1


def test_mutating_number_intrinsics_cannot_admit_nonfinite_json(runtime):
    runner, _, _ = runtime
    with pytest.raises(CodeModeError):
        runner.execute("async()=>{Number.isFinite=()=>true;return Infinity;}")


def test_caught_local_serialization_error_does_not_skip_the_next_rpc_id(runtime):
    runner, router, predictor = runtime
    assert runner.execute("""async input => {
      try { await veya.predict({...input,state:{bad:()=>true}}); } catch(_) {}
      return (await veya.predict(input)).answers.urgent.noul;
    }""", request(), router=router) == .51
    assert predictor.calls == 1


def test_argument_getter_cannot_reenter_and_reorder_host_calls(runtime):
    runner, router, predictor = runtime
    assert runner.execute("""async input => {
      const bad = {...input, get state() { return veya.predict(input); }};
      try { await veya.predict(bad); } catch(_) {}
      return (await veya.predict(input)).answers.urgent.noul;
    }""", request(), router=router) == .51
    assert predictor.calls == 1


def test_initial_expression_cannot_replace_private_promise_chaining(runtime):
    runner, _, _ = runtime
    assert runner.execute("""(() => {
      Promise.prototype.then = () => Promise.resolve(42);
      Promise.resolve = () => ({then:()=>42});
      return async () => 99;
    })()""") == 99


def test_foreign_runtime_public_agent_does_not_bypass_local_hook_namespace(runtime, monkeypatch):
    runner, router, _ = runtime
    class ForeignAgent(ONNXAgent):
        pass
    foreign = object.__new__(ForeignAgent)
    foreign.cfg, foreign.hooks = Predictor.cfg, ()
    foreign.system_one = lambda *args, **kwargs: pytest.fail("foreign hook namespace must not enter inference")
    router.attach("english", foreign)
    monkeypatch.setitem(__import__("sys").modules, "laya_onnx", SimpleNamespace(ONNXAgent=ForeignAgent))
    with pytest.raises(CodeModeError) as error:
        runner.execute("async input => await veya.predict(input)", request(), router=router)
    assert error.value.code == "code_resource_limit"


def test_concurrent_close_settles_guest_before_returning_without_a_late_success(runtime, monkeypatch):
    _, router, _ = runtime
    runner = CodeModeRunner(CodeModeLimits(wall_seconds=10, cpu_seconds=5))
    entered = threading.Event()
    outcomes = []
    before = {child.pid for child in multiprocessing.active_children()}
    def host(*args, **kwargs):
        entered.set()
        return {}
    monkeypatch.setattr(runner, "_invoke", host)
    def execute():
        try:
            outcomes.append(runner.execute("async()=>{await veya.status({});while(true){};return 42}", router=router))
        except BaseException as error:
            outcomes.append(error)
    thread = threading.Thread(target=execute)
    thread.start()
    assert entered.wait(4)
    time.sleep(.03)
    started = time.monotonic()
    runner.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], CodeModeError)
    assert outcomes[0].code == "code_unavailable"
    assert time.monotonic() - started < 2
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_close_wins_race_before_atomic_admission_without_spawning_a_guest():
    runner = CodeModeRunner()
    validating, release = threading.Event(), threading.Event()
    outcomes = []
    class Paused(dict):
        def items(self):
            validating.set()
            assert release.wait(3)
            return super().items()
    def execute():
        try:
            outcomes.append(runner.execute("async()=>42", Paused(value=1)))
        except BaseException as error:
            outcomes.append(error)
    before = {child.pid for child in multiprocessing.active_children()}
    thread = threading.Thread(target=execute)
    thread.start()
    assert validating.wait(2)
    runner.close()
    release.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], CodeModeError)
    assert outcomes[0].code == "code_unavailable"
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_reentrant_close_from_host_callback_is_refused_without_deadlock(runtime, monkeypatch):
    runner, router, _ = runtime
    def host(*args, **kwargs):
        with pytest.raises(RuntimeError, match="outside"):
            runner.close()
        return {"still_open": True}
    monkeypatch.setattr(runner, "_invoke", host)
    assert runner.execute("async()=>await veya.status({})", router=router) == {"still_open": True}
    assert runner.execute("async()=>42") == 42


def test_reentrant_close_from_active_caller_is_refused_without_deadlock(runtime, monkeypatch):
    import psutil
    runner, _, _ = runtime
    original = psutil.Process
    checks = []
    class Monitor:
        def __init__(self, pid):
            self.process = original(pid)
        def cpu_times(self):
            with pytest.raises(RuntimeError, match="outside"):
                runner.close()
            checks.append(True)
            return self.process.cpu_times()
    monkeypatch.setattr(psutil, "Process", Monitor)
    assert runner.execute("async()=>42") == 42
    assert checks


def test_failed_process_start_recovers_the_admission_permit(runtime, monkeypatch):
    runner, _, _ = runtime
    process_type = multiprocessing.get_context("spawn").Process
    original = process_type.start
    def fail(*args, **kwargs):
        raise OSError("private-process-launch-path")
    monkeypatch.setattr(process_type, "start", fail)
    with pytest.raises(CodeModeError) as error:
        runner.execute("async()=>42")
    assert "private-process-launch-path" not in str(error.value)
    monkeypatch.setattr(process_type, "start", original)
    assert runner.execute("async()=>42") == 42


def test_ready_monitor_can_observe_a_short_guest_that_already_exited(runtime, monkeypatch):
    import psutil
    runner, _, _ = runtime
    original = psutil.Process
    class Monitor:
        def __init__(self, pid):
            self.process, self.pid, self.first = original(pid), pid, True
        def cpu_times(self):
            if self.first:
                self.first = False
                guest = next((child for child in multiprocessing.active_children() if child.pid == self.pid), None)
                if guest is not None:
                    guest.join(timeout=3)
                    assert not guest.is_alive()
                raise psutil.NoSuchProcess(self.pid)
            return self.process.cpu_times()
    monkeypatch.setattr(psutil, "Process", Monitor)
    assert runner.execute("async()=>42") == 42


def test_cpu_monitor_constructor_can_race_with_a_completed_guest(runtime, monkeypatch):
    import psutil
    runner, _, _ = runtime
    def gone(pid):
        guest = next((child for child in multiprocessing.active_children() if child.pid == pid), None)
        if guest is not None:
            guest.join(timeout=3)
            assert not guest.is_alive()
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", gone)
    assert runner.execute("async()=>42") == 42


def test_missing_cpu_monitor_for_a_live_guest_fails_closed_and_recovers(runtime, monkeypatch):
    import psutil
    runner, _, _ = runtime
    original = psutil.Process
    def unavailable(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", unavailable)
    with pytest.raises(CodeModeError) as error:
        runner.execute("async()=>{while(true){}}")
    assert error.value.code == "code_sandbox_failed"
    monkeypatch.setattr(psutil, "Process", original)
    assert runner.execute("async()=>42") == 42
