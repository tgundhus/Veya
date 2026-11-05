"""Required decision storage must fail closed; run without checkpoints or network."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya import DecisionCache, Router  # noqa: E402
from laya.consistency import _MemoryStore  # noqa: E402
from laya.errors import DecisionCacheError  # noqa: E402
from laya.hooks import AsyncHook, PredictContext, dispatch  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
import laya.consistency as consistency  # noqa: E402


QUESTIONS = {"keep": {"type": "noul", "instructions": "Keep this?"}}


class Agent:
    model_id = "test/checkpoint"
    revision = "one"
    cfg = {"max_len": 512, "head_max_len": 192}

    def __init__(self, value=0.51):
        self.value = value
        self.calls = 0

    def predict_batch(self, states, questions, **kwargs):
        self.calls += 1
        return [{"answers": {name: {"type": "noul", "noul": self.value} for name in questions},
                 "usage": {"input_tokens": 1, "output_tokens": 0}} for _ in states]

    def system_one(self, state, questions, **kwargs):
        return self.predict_batch([state], questions, **kwargs)[0]


def router(cache, agent=None, hooks=()):
    instance = Router(hooks=[cache, *hooks], hooks_raise=False)
    agent = agent or Agent()
    instance.attach("english", agent)
    return instance, agent


class Store(_MemoryStore):
    def __init__(self, fail=None):
        super().__init__(None)
        self.fail = fail

    def get(self, *args):
        if self.fail == "read":
            raise OSError("private database path")
        return super().get(*args)

    def add(self, *args):
        if self.fail == "write":
            raise OSError("private database path")
        result = super().add(*args)
        if self.fail == "short_write":
            return result[:-1]
        return result

    def renew(self, *args):
        if self.fail == "renew":
            raise OSError("private database path")
        return super().renew(*args)


class RequiredCacheTests(unittest.TestCase):
    def test_required_is_a_strict_boolean(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                DecisionCache(required=value)

    def test_read_write_and_corruption_never_return_an_alternative(self):
        for failure in ("read", "write", "short_write", "corrupt"):
            store = Store(None if failure == "corrupt" else failure)
            cache = DecisionCache(store=store, required=True)
            instance, agent = router(cache)
            if failure == "corrupt":
                instance.predict("hello", QUESTIONS)
                store._data[next(iter(store._data))] = consistency._ENTRY.pack(time.time(), float("inf")) + b"bad"
                agent.value = 0.49
            with self.subTest(failure=failure), self.assertRaises(DecisionCacheError) as caught:
                instance.predict("hello", QUESTIONS, hooks_raise=False)
            self.assertNotIn("private database path", str(caught.exception))
            self.assertEqual(agent.calls, 0 if failure == "read" else 1)
            self.assertEqual(cache._inflight, {})
            self.assertFalse(cache._pending)

    def test_required_rejects_nonretained_policy_and_nonserializable_results(self):
        for lifetime in (lambda *_: 0, lambda *_: -1, lambda *_: "invalid"):
            cache = DecisionCache(required=True, ttl=lifetime)
            instance, _ = router(cache)
            with self.assertRaises(DecisionCacheError):
                instance.predict("hello", QUESTIONS)
            self.assertEqual(cache._inflight, {})
        class Nonserializable(Agent):
            def predict_batch(self, *args, **kwargs):
                result = super().predict_batch(*args, **kwargs)
                result[0]["extra"] = object()
                return result
        cache = DecisionCache(required=True)
        with self.assertRaises(DecisionCacheError):
            router(cache, Nonserializable())[0].predict("hello", QUESTIONS)
        self.assertEqual(cache._inflight, {})

    def test_partial_commit_or_capacity_eviction_is_not_a_success(self):
        cache = DecisionCache(maxsize=1, required=True)
        ctx = PredictContext(states=["a", "b"], questions=QUESTIONS, model="english", agent=Agent())
        dispatch([cache], "on_predict_start", ctx, raise_errors=False)
        ctx.results = Agent().predict_batch(ctx.states, QUESTIONS)
        with self.assertRaises(DecisionCacheError):
            dispatch([cache], "on_predict_end", ctx, raise_errors=False)
        self.assertEqual(cache._inflight, {})

    def test_commit_confirmation_read_failure_is_fatal(self):
        class LoseRead(Store):
            def add(self, *args):
                held = super().add(*args)
                self.fail = "read"
                return held
        cache = DecisionCache(store=LoseRead(), required=True)
        with self.assertRaises(DecisionCacheError):
            router(cache)[0].predict("a", QUESTIONS)
        self.assertEqual(cache._inflight, {})

    def test_empty_question_results_are_still_authorized(self):
        cache = DecisionCache(required=True)
        instance, agent = router(cache)
        first = instance.predict("a", {})
        self.assertEqual(instance.predict("a", {})["answers"], first["answers"])
        self.assertEqual(agent.calls, 1)
        ctx = PredictContext(states=[], questions=QUESTIONS, agent=Agent())
        dispatch([cache], "on_predict_start", ctx, raise_errors=False)
        self.assertEqual(ctx.results, [])
        ctx.results = Agent().predict_batch(["invented"], QUESTIONS)
        with self.assertRaises(DecisionCacheError):
            dispatch([cache], "on_predict_end", ctx, raise_errors=False)

    def test_direct_language_sensitive_agent_cannot_bypass_required_storage(self):
        agent = Agent()
        agent.lang_temperatures = {"de": {"temperature": [1, 1, 1]}}
        ctx = PredictContext(states=["a"], questions=QUESTIONS, agent=agent)
        cache = DecisionCache(required=True)
        with self.assertRaisesRegex(DecisionCacheError, "Router"):
            dispatch([cache], "on_predict_start", ctx, raise_errors=False)

    def test_incomplete_and_nonobject_results_do_not_pass_as_commits(self):
        for results in (None, [], [None], ["invalid"], [{}], [{}, {}]):
            cache = DecisionCache(required=True)
            ctx = PredictContext(states=["a"], questions=QUESTIONS, model="english", agent=Agent())
            dispatch([cache], "on_predict_start", ctx, raise_errors=False)
            ctx.results = results
            with self.subTest(results=results), self.assertRaises(DecisionCacheError):
                dispatch([cache], "on_predict_end", ctx, raise_errors=False)
            self.assertEqual(cache._inflight, {})

    def test_async_adapter_preserves_required_ordering_and_failure_policy(self):
        cache = DecisionCache(store=Store("read"), required=True)
        ctx = PredictContext(states=["a"], questions=QUESTIONS, model="english", agent=Agent())
        with self.assertRaises(DecisionCacheError):
            dispatch([AsyncHook(cache)], "on_predict_start", ctx, raise_errors=False)
        class Transform:
            def on_predict_end(self, ctx):
                ctx.results[0]["answers"]["keep"]["noul"] += 0.1
        cache = DecisionCache(required=True)
        instance = Router(hooks=[AsyncHook(cache), Transform()], hooks_raise=False)
        instance.attach("english", Agent())
        first = instance.predict("a", QUESTIONS)
        self.assertEqual(instance.predict("a", QUESTIONS)["answers"], first["answers"])

    def test_renew_failure_is_fatal_and_default_mode_is_unchanged(self):
        store = Store()
        cache = DecisionCache(store=store, ttl=160, renew_on_hit=True, required=True)
        instance, _ = router(cache)
        with patch.object(consistency, "_now", return_value=1000):
            instance.predict("hello", QUESTIONS)
        store.fail = "renew"
        with patch.object(consistency, "_now", return_value=1020), self.assertRaises(DecisionCacheError):
            instance.predict("hello", QUESTIONS)
        default, agent = router(DecisionCache(store=Store("read")))
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(default.predict("hello", QUESTIONS)["answers"]["keep"]["noul"], 0.51)
        self.assertEqual(agent.calls, 1)

    def test_terminal_hooks_commit_transformations_and_restore_stored_hits(self):
        class Transform:
            def on_predict_start(self, ctx):
                ctx.states = [state.lower() for state in ctx.states]

            def on_predict_end(self, ctx):
                if ctx.error is None and ctx.results:
                    ctx.results[0]["answers"]["keep"]["noul"] += 0.1
        cache = DecisionCache(required=True)
        instance, agent = router(cache, hooks=[Transform()])
        first = instance.predict("HELLO", QUESTIONS)
        second = instance.predict("hello", QUESTIONS)
        self.assertEqual(first["answers"], second["answers"])
        self.assertAlmostEqual(second["answers"]["keep"]["noul"], 0.61)
        self.assertEqual(agent.calls, 1)

    def test_required_cache_rejects_prior_skip_multiple_caches_and_identity_mutation(self):
        class Skip:
            def on_predict_start(self, ctx):
                ctx.skip(Agent().predict_batch(ctx.states, QUESTIONS))
        with self.assertRaises(DecisionCacheError):
            router(DecisionCache(required=True), hooks=[Skip()])[0].predict("a", QUESTIONS)
        with self.assertRaises(DecisionCacheError):
            router(DecisionCache(required=True), hooks=[DecisionCache(required=True)])[0].predict("a", QUESTIONS)
        class ChangeIdentity:
            def on_predict_end(self, ctx):
                ctx.questions = {"changed": {"type": "noul", "instructions": "Changed policy"}}
        cache = DecisionCache(required=True)
        with self.assertRaises(DecisionCacheError):
            router(cache, hooks=[ChangeIdentity()])[0].predict("a", QUESTIONS)
        self.assertEqual(cache._inflight, {})

    def test_all_rows_and_duplicates_use_the_committed_winner(self):
        cache = DecisionCache(required=True)
        instance, agent = router(cache)
        instance.predict("a", QUESTIONS)
        agent.value = 0.49
        result = instance.predict_batch([{"state": state, "questions": QUESTIONS}
                                         for state in ("a", "b", "b")], hooks_raise=False)
        self.assertEqual([r["answers"]["keep"]["noul"] for r in result], [0.51, 0.49, 0.49])
        self.assertEqual(cache._inflight, {})

    def test_coalesce_deadline_fails_without_displacing_the_owner(self):
        cache = DecisionCache(required=True)
        ctx = PredictContext(states=["hello"], questions=QUESTIONS, model="english", agent=Agent())
        key = cache._keys(ctx, ctx.states, False)[0]
        owner = (threading.get_ident() + 1, threading.Event())
        cache._inflight[key] = owner
        with patch.object(consistency, "_COALESCE_WAIT", 0), self.assertRaises(DecisionCacheError):
            dispatch([cache], "on_predict_start", ctx, raise_errors=False)
        self.assertIs(cache._inflight[key], owner)
        cache._release([(key, owner)])

    def test_start_and_end_deadlines_are_unsuppressible(self):
        for method in ("get", "add"):
            entered, release = threading.Event(), threading.Event()
            class Blocking(Store):
                def get(self, *args):
                    if method == "get":
                        entered.set()
                        release.wait(2)
                    return super().get(*args)

                def add(self, *args):
                    if method == "add":
                        entered.set()
                        release.wait(2)
                    return super().add(*args)
            cache = DecisionCache(store=Blocking(), required=True)
            instance, _ = router(cache)
            try:
                with self.assertRaises(DecisionCacheError):
                    instance.predict("hello", QUESTIONS, hooks_raise=False, hooks_timeout=0.01)
                self.assertTrue(entered.is_set())
            finally:
                release.set()
            deadline = time.monotonic() + 2
            while cache._inflight and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(cache._inflight)

    def test_ordinary_prediction_hook_deadlines_also_fail_closed(self):
        for event in ("on_predict_start", "on_predict_end"):
            entered, release, done = threading.Event(), threading.Event(), threading.Event()
            def late(ctx):
                entered.set()
                release.wait(2)
                if ctx.results:
                    ctx.results[0]["answers"]["keep"]["noul"] = 0.49
                done.set()
            hook = type("Late", (), {})()
            setattr(hook, event, late)
            cache = DecisionCache(required=True)
            instance, _ = router(cache, hooks=[hook])
            try:
                with self.subTest(event=event), self.assertRaises(DecisionCacheError):
                    instance.predict("hello", QUESTIONS, hooks_raise=False, hooks_timeout=0.01)
                self.assertTrue(entered.is_set())
            finally:
                release.set()
                self.assertTrue(done.wait(2))

    def test_new_commit_is_returned_from_immutable_authoritative_bytes(self):
        class Capturing(Agent):
            def predict_batch(self, *args, **kwargs):
                self.last = super().predict_batch(*args, **kwargs)
                return self.last
        agent = Capturing()
        class MutatingStore(Store):
            def add(self, *args):
                agent.last[0]["answers"]["keep"]["noul"] = 0.49
                return super().add(*args)
        cache = DecisionCache(store=MutatingStore(), required=True)
        instance, _ = router(cache, agent)
        first = instance.predict("hello", QUESTIONS)
        self.assertEqual(first["answers"]["keep"]["noul"], 0.51)
        self.assertEqual(instance.predict("hello", QUESTIONS)["answers"], first["answers"])

    def test_prediction_hooks_cannot_mark_an_error_and_bypass_authorization(self):
        def mark_error(ctx):
            ctx.error = RuntimeError("telemetry failed")
        for event in ("on_predict_start", "on_predict_end"):
            hook = type("MarkError", (), {})()
            setattr(hook, event, mark_error)
            cache = DecisionCache(required=True)
            instance, agent = router(cache, hooks=[hook])
            for value in (0.51, 0.49):
                agent.value = value
                with self.subTest(event=event, value=value), self.assertRaises(DecisionCacheError):
                    instance.predict("hello", QUESTIONS, hooks_raise=False)
                self.assertEqual(cache.cache_info()["size"], 0)
                self.assertFalse(cache._inflight)

    def test_required_agent_cache_is_direct_only_and_cannot_bypass_router_hooks(self):
        cache = DecisionCache(required=True)
        predictor = Agent()
        agent = ONNXAgent.__new__(ONNXAgent)
        agent.model_id = "test/checkpoint"
        agent.cfg = {"max_len": 512, "head_max_len": 192}
        agent.hooks = [AsyncHook(cache)]
        agent._infer_batch = lambda states, questions, **kwargs: predictor.predict_batch(states, questions)
        direct = agent.system_one("direct", QUESTIONS)
        self.assertEqual(agent.system_one("direct", QUESTIONS)["answers"], direct["answers"])
        self.assertEqual(predictor.calls, 1)

        instance = Router(hooks_raise=False)
        instance.attach("english", agent)
        calls = (lambda: instance.predict("nested", QUESTIONS),
                 lambda: instance.predict_batch([{"state": "nested", "questions": QUESTIONS}]),
                 lambda: instance.predict_long("nested", QUESTIONS))
        for call in calls:
            with self.assertRaisesRegex(DecisionCacheError, "Router"):
                call()
        self.assertEqual(predictor.calls, 1)

    def test_router_still_accepts_agent_like_objects_with_no_hooks(self):
        agent = Agent()
        agent.hooks = None
        instance, _ = router(DecisionCache(), agent)
        self.assertEqual(instance.predict("hello", QUESTIONS)["answers"]["keep"]["noul"], 0.51)
        self.assertEqual(instance.predict_batch([{"state": "batch", "questions": QUESTIONS}])[0]
                         ["answers"]["keep"]["noul"], 0.51)

    def test_sqlite_full_sync_and_restart_first_writer_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "decisions.sqlite")
            cache = DecisionCache(path, required=True, maxsize=None, fingerprint="policy-v1")
            self.assertEqual(cache._store._db.execute("PRAGMA synchronous").fetchone()[0], 2)
            first = router(cache, Agent(0.51))[0].predict("hello", QUESTIONS)
            cache.close()
            cache = DecisionCache(path, required=True, maxsize=None, fingerprint="policy-v1")
            instance, agent = router(cache, Agent(0.49))
            self.assertEqual(instance.predict("hello", QUESTIONS)["answers"], first["answers"])
            self.assertEqual(agent.calls, 0)
            cache.close()
            normal = DecisionCache(path)
            self.assertEqual(normal._store._db.execute("PRAGMA synchronous").fetchone()[0], 1)
            normal.close()

    def test_separate_sqlite_writers_return_the_same_first_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "shared.sqlite")
            first = DecisionCache(path, required=True, maxsize=None, fingerprint="same-policy")
            second = DecisionCache(path, required=True, maxsize=None, fingerprint="same-policy")
            a, _ = router(first, Agent(0.51))
            class Race(Agent):
                def system_one(self, *args, **kwargs):
                    a.predict("hello", QUESTIONS)
                    return super().system_one(*args, **kwargs)
            b, _ = router(second, Race(0.49))
            self.assertEqual(b.predict("hello", QUESTIONS)["answers"]["keep"]["noul"], 0.51)
            first.close()
            second.close()

    def test_independent_process_writers_agree_on_the_stored_decision(self):
        program = """
import runpy, sys, time
from pathlib import Path
scope = runpy.run_path(sys.argv[1])
cache = scope['DecisionCache'](sys.argv[2], required=True, maxsize=None, fingerprint='process-policy')
value = float(sys.argv[3])
directory = Path(sys.argv[4])
class Slow(scope['Agent']):
    def system_one(self, *args, **kwargs):
        (directory / ('ready-' + str(value))).touch()
        deadline = time.monotonic() + 5
        while len(list(directory.glob('ready-*'))) < 2:
            if time.monotonic() >= deadline:
                raise RuntimeError('other writer never reached inference')
            time.sleep(0.01)
        return super().system_one(*args, **kwargs)
instance, _ = scope['router'](cache, Slow(value))
print(instance.predict('hello', scope['QUESTIONS'])['answers']['keep']['noul'])
cache.close()
"""
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "shared.sqlite")
            DecisionCache(path, required=True).close()
            processes = [subprocess.Popen([sys.executable, "-c", program, str(Path(__file__).resolve()),
                                           path, value, directory], cwd=ROOT, env=os.environ.copy(),
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                         for value in ("0.51", "0.49")]
            try:
                outputs = [process.communicate(timeout=10) for process in processes]
                self.assertEqual([p.returncode for p in processes], [0, 0], outputs)
                winners = [float(stdout.strip()) for stdout, _ in outputs]
                self.assertEqual(winners[0], winners[1])
                self.assertIn(winners[0], (0.51, 0.49))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()


if __name__ == "__main__":
    unittest.main()
