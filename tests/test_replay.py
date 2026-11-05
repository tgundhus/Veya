"""Verified model-free replay, store identity and conservative eligibility regressions."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sqlite3
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
from laya.agent import Agent  # noqa: E402
from laya.checkpoint import _checkpoint_signature  # noqa: E402
from laya.errors import DecisionCacheError  # noqa: E402
from laya.hooks import clear_default_hooks, set_default_hooks  # noqa: E402
from laya.replay import load_manifest, _onnx_artifacts  # noqa: E402


QUESTIONS = {"keep": {"type": "noul", "instructions": "Keep this?"}}


def fixture(directory, *, required=True, ttl=None, renew_on_hit=False, lang_temperatures=None):
    """A built-in Agent with controlled methods; actual-checkpoint benchmark is separate."""
    directory = Path(directory)
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    for name in ("rl_agent_config.json", "model.safetensors"):
        if not (checkpoint / name).exists():
            (checkpoint / name).write_text("{}")
    agent = object.__new__(Agent)
    agent.model_id, agent.subfolder, agent.revision = str(checkpoint), None, None
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    agent.temperature, agent.temperature_by_options = [1.0, 1.0, 1.0], {2: 1.2}
    agent.lang_temperatures = lang_temperatures or {}
    agent.binning_map, agent.hooks = None, []
    agent._checkpoint_signature = _checkpoint_signature(str(checkpoint), str(checkpoint / "model.safetensors"), {})
    agent.calls = 0

    def answer(state, questions, **kwargs):
        agent.calls += 1
        return {"answers": {key: {"type": "noul", "noul": 0.51, "answer_confidence": 0.51}
                            for key in questions}, "usage": {"input_tokens": 1, "output_tokens": 0}}

    agent.system_one = answer
    agent.predict_batch = lambda states, questions, **kwargs: [answer(state, questions, **kwargs) for state in states]
    cache = DecisionCache(str(directory / "decisions.sqlite"), required=required, ttl=ttl,
                          renew_on_hit=renew_on_hit, maxsize=None)
    router = Router(models={"english": str(checkpoint)}, hooks=[cache], performance_mode="default")
    router._agents["english"] = agent
    return router, cache, agent


def restarted(directory, cache, manifest, **kwargs):
    return Router(models={"english": str(Path(directory) / "checkpoint")}, hooks=[cache],
                  replay_manifest=str(manifest), performance_mode="default", **kwargs)


class ReplayTests(unittest.TestCase):
    def tearDown(self):
        clear_default_hooks()

    def test_exact_cold_single_and_batch_hits_never_load_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            first = router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            exported = router.export_replay_manifest(manifest)
            self.assertEqual(exported["store_id"], cache.store_id)
            cold = restarted(directory, cache, manifest)
            info = cache.cache_info()
            self.assertTrue(cold.validate_replay_manifest())
            self.assertEqual(cache.cache_info(), info)
            with patch.object(cold, "load", side_effect=AssertionError("model was loaded")):
                replay = cold.predict("hello", QUESTIONS, model="english")
                batch = cold.predict_batch([{"state": "hello", "questions": QUESTIONS, "model": "english"}] * 2)
            self.assertEqual(first["answers"], replay["answers"])
            self.assertTrue(all(item["answers"] == first["answers"] for item in batch))
            self.assertEqual(cache.cache_info()["hits"], 3)
            self.assertFalse(cache._pending)
            self.assertEqual(cache._inflight, {})
            cache.close()

    def test_miss_is_read_only_and_mixed_batch_only_loads_for_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            before = cache.cache_info()
            self.assertIsNone(cold.try_replay("new", QUESTIONS, model="english"))
            self.assertEqual(cache.cache_info(), before)
            self.assertFalse(cache._pending)
            self.assertEqual(cache._inflight, {})
            with patch.object(cold, "load", return_value=agent) as load:
                results = cold.predict_batch([{"state": state, "questions": QUESTIONS, "model": "english"}
                                              for state in ("hello", "new")])
            self.assertEqual(len(results), 2)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(agent.calls, 2)
            cache.close()

    def test_budget_language_and_confidence_contracts_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory, lang_temperatures={"de": {"temperature": [1, 1, 1]}})
            first = router.predict("hello", QUESTIONS, model="english", lang="de", max_len=123)
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            self.assertIsNone(cold.try_replay("hello", QUESTIONS, model="english", lang="en", max_len=123))
            self.assertIsNone(cold.try_replay("hello", QUESTIONS, model="english", lang="de", max_len=124))
            replay = cold.try_replay("hello", QUESTIONS, model="english", lang="de", max_len=123, min_confidence=0.8)
            self.assertEqual(replay["answers"]["keep"]["noul"], first["answers"]["keep"]["noul"])
            self.assertTrue(replay["answers"]["keep"]["low_confidence"])
            self.assertEqual(replay["routing"], first["routing"])
            cache.close()

    def test_ttl_renewal_and_expiry_follow_existing_cache_policy(self):
        with tempfile.TemporaryDirectory() as directory, patch("laya.consistency._now", return_value=100):
            router, cache, _ = fixture(directory, ttl=16, renew_on_hit=True)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            with patch("laya.consistency._now", return_value=108):
                self.assertIsNotNone(cold.try_replay("hello", QUESTIONS, model="english"))
            with patch("laya.consistency._now", return_value=123):
                self.assertIsNotNone(cold.try_replay("hello", QUESTIONS, model="english"))
            with patch("laya.consistency._now", return_value=140):
                self.assertIsNone(cold.try_replay("hello", QUESTIONS, model="english"))
            cache.close()

    def test_store_read_decode_and_renew_errors_stay_terminal(self):
        for operation in ("read", "decode", "renew"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                router, cache, _ = fixture(directory, ttl=16, renew_on_hit=True)
                with patch("laya.consistency._now", return_value=100):
                    router.predict("hello", QUESTIONS, model="english")
                manifest = Path(directory) / "replay.json"
                router.export_replay_manifest(manifest)
                cold = restarted(directory, cache, manifest)
                method = "renew" if operation == "renew" else "get"
                kwargs = {"return_value": (b"corrupt", 116)} if operation == "decode" else {"side_effect": OSError("private path")}
                with patch("laya.consistency._now", return_value=108), patch.object(cache._store, method, **kwargs):
                    with self.assertRaises(DecisionCacheError):
                        cold.try_replay("hello", QUESTIONS, model="english")
                cache.close()

    def test_custom_hooks_and_deadlines_keep_original_prediction_path(self):
        class Hook:
            def on_predict_start(self, ctx):
                pass
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            set_default_hooks([Hook()])
            self.assertFalse(cold.validate_replay_manifest())
            self.assertIsNone(cold.try_replay("hello", QUESTIONS, model="english"))
            clear_default_hooks()
            cold.hooks_timeout = 1
            self.assertIsNone(cold.try_replay("hello", QUESTIONS, model="english"))
            cold.hooks_timeout = None
            with patch.object(cold, "load", return_value=agent) as load:
                cold.predict("hello", QUESTIONS, model="english", hooks=[Hook()])
                self.assertEqual(load.call_count, 1)
            cache.close()

    def test_invalid_routing_controls_keep_their_input_error_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            for controls in ({"task": "missing-task"}, {"model": "missing-model"}):
                with self.subTest(controls=controls), self.assertRaises(ValueError):
                    cold.predict("hello", QUESTIONS, **controls)
            cache.close()

    def test_serialized_fixed_retention_hits_respect_the_hook_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest, hooks_concurrent=False)
            entered = threading.Event()
            lookup = cache._lookup_replay
            def record(*args, **kwargs):
                entered.set()
                return lookup(*args, **kwargs)
            result = []
            with patch.object(cache, "_lookup_replay", side_effect=record):
                with cold._hooks_lock:
                    worker = threading.Thread(target=lambda: result.append(cold.try_replay("hello", QUESTIONS, model="english")))
                    worker.start()
                    self.assertFalse(entered.wait(0.03))
                worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result[0]["answers"]["keep"]["noul"], 0.51)
            cache.close()

    def test_custom_retention_keeps_ordinary_suppression_and_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory, required=False, ttl=16, renew_on_hit=True)
            router.hooks_raise = False
            with patch("laya.consistency._now", return_value=100):
                router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest, hooks_raise=False, hooks_concurrent=False)
            cold._agents["english"] = agent
            cache.ttl = lambda *_: (_ for _ in ()).throw(ValueError("custom policy failed"))
            self.assertFalse(cold.validate_replay_manifest())
            with patch("laya.consistency._now", return_value=108), self.assertWarns(RuntimeWarning):
                self.assertEqual(cold.predict("hello", QUESTIONS, model="english")["answers"]["keep"]["noul"], 0.51)
            active = maximum = 0
            mutex = threading.Lock()
            def policy(*_):
                nonlocal active, maximum
                with mutex:
                    active += 1
                    maximum = max(active, maximum)
                time.sleep(0.02)
                with mutex:
                    active -= 1
                return 16
            cache.ttl = policy
            with patch("laya.consistency._now", return_value=108):
                workers = [threading.Thread(target=lambda: cold.predict("hello", QUESTIONS, model="english")) for _ in range(2)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(2)
                    self.assertFalse(worker.is_alive())
            self.assertEqual(maximum, 1)
            cache.close()

    def test_changed_configuration_files_or_resident_metadata_refuse_replay(self):
        for change in ("configuration", "file", "resident"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                router, cache, agent = fixture(directory)
                router.predict("hello", QUESTIONS, model="english")
                manifest = Path(directory) / "replay.json"
                router.export_replay_manifest(manifest)
                cold = restarted(directory, cache, manifest)
                if change == "configuration":
                    cold.auto_task_detection = True
                elif change == "file":
                    (Path(directory) / "checkpoint" / "rl_agent_config.json").write_text('{"changed":true}')
                else:
                    cold._agents["english"] = agent
                    agent.temperature[0] = 2
                with self.assertRaises(DecisionCacheError):
                    cold.predict("hello", QUESTIONS, model="english")
                cache.close()

    def test_corrupt_incompatible_and_missing_manifests_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            good = manifest.read_bytes()
            variants = [b"{}", good.replace(b'veya-replay-v1', b'veya-replay-v9'),
                        good.replace(b'"library"', b'"librarx"'), good + b"invalid"]
            for content in variants:
                manifest.write_bytes(content)
                with self.assertRaises(DecisionCacheError):
                    restarted(directory, cache, manifest)
            with self.assertRaises(DecisionCacheError):
                restarted(directory, cache, Path(directory) / "missing.json")
            cache.close()

    def test_manifest_is_bound_to_its_store_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            other = DecisionCache(str(Path(directory) / "other.sqlite"), required=True)
            cold = restarted(directory, other, manifest)
            with self.assertRaises(DecisionCacheError):
                cold.try_replay("hello", QUESTIONS, model="english")
            other.close()
            cache.fingerprint = "different-policy"
            with self.assertRaises(DecisionCacheError):
                router.try_replay("hello", QUESTIONS, model="english")
            cache.close()

    def test_empty_compose_environment_matches_unset_revision_and_digests(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            for name in ("LAYA_REVISION", "LAYA_SHA256_DIGESTS"):
                os.environ.pop(name, None)
            router, cache, _ = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            os.environ["LAYA_REVISION"] = "  "
            os.environ["LAYA_SHA256_DIGESTS"] = ""
            cold = restarted(directory, cache, manifest)
            self.assertIsNotNone(cold.try_replay("hello", QUESTIONS, model="english"))
            cache.close()

    def test_concurrent_replays_return_independent_authoritative_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            values = []
            workers = [threading.Thread(target=lambda: values.append(cold.try_replay("hello", QUESTIONS, model="english")))
                       for _ in range(12)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            self.assertEqual(len(values), 12)
            values[0]["answers"]["keep"]["noul"] = 0.49
            self.assertTrue(all(value["answers"]["keep"]["noul"] == 0.51 for value in values[1:]))
            cache.close()

    def test_process_restart_replays_without_importing_inference_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            router.agent_kwargs = {"lang_temperatures": {}}
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            store_id = cache.store_id
            cache.close()
            code = """
import importlib.abc,json,sys
attempted=[]
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','transformers','onnxruntime','tokenizers','onnx'):
   attempted.append(name)
   raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from laya import Router,DecisionCache
cache=DecisionCache(sys.argv[1],required=True,expected_store_id=sys.argv[4])
router=Router(models={'english':sys.argv[2]},hooks=[cache],replay_manifest=sys.argv[3],
              agent_kwargs={'lang_temperatures':{}},performance_mode=2)
assert router.validate_replay_manifest()
result=router.predict('hello',{'keep':{'type':'noul','instructions':'Keep this?'}},model='english')
print(result['answers']['keep']['noul'])
router.unload()
assert not attempted,attempted
cache.close()
"""
            result = subprocess.run([sys.executable, "-c", code, str(Path(directory) / "decisions.sqlite"),
                                     str(Path(directory) / "checkpoint"), str(manifest), store_id],
                                    cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "0.51")

    def test_profile_threads_are_deferred_until_a_real_model_is_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            with patch("laya.router._apply_torch_threads") as apply:
                cold = Router(models={"english": str(Path(directory) / "checkpoint")}, hooks=[cache],
                              replay_manifest=str(manifest), performance_mode=2)
                apply.assert_not_called()
                self.assertIsNotNone(cold.try_replay("hello", QUESTIONS, model="english"))
                apply.assert_not_called()
                with patch.object(cold, "_build", return_value=agent):
                    cold.predict("new", QUESTIONS, model="english")
                    cold.predict("newer", QUESTIONS, model="english")
                self.assertEqual(apply.call_count, 1)
            cache.close()

    def test_deferred_runtime_callback_runs_once_outside_the_metadata_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            calls = []
            def setup():
                checks = []
                worker = threading.Thread(target=lambda: checks.append(cold.validate_replay_manifest()))
                worker.start()
                worker.join(1)
                self.assertFalse(worker.is_alive(), "runtime setup held the Router metadata lock")
                self.assertEqual(checks, [True])
                calls.append(True)
            cold._replay_runtime_setup = setup
            cold.predict("hello", QUESTIONS, model="english")
            self.assertEqual(calls, [])
            with patch.object(cold, "_build", return_value=agent):
                cold.predict("new", QUESTIONS, model="english")
                cold.predict("newer", QUESTIONS, model="english")
            self.assertEqual(calls, [True])
            self.assertIsNone(cold._replay_runtime_setup)
            cache.close()

    def test_preparation_rechecks_strong_digest_entries_before_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            for entry in agent._checkpoint_signature:
                path = Path(agent.model_id) / entry[0]
                entry[1:] = ["sha256", hashlib.sha256(path.read_bytes()).hexdigest()]
            (Path(agent.model_id) / "rl_agent_config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "verified digest"):
                router.export_replay_manifest(Path(directory) / "replay.json")
            cache.close()

    def test_missing_or_memory_cache_cannot_authorize_a_configured_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            for hooks in ([], [DecisionCache(required=True)]):
                invalid = Router(models={"english": str(Path(directory) / "checkpoint")}, hooks=hooks,
                                 replay_manifest=str(manifest), performance_mode="default")
                with self.assertRaises(DecisionCacheError):
                    invalid.validate_replay_manifest()
            cache.close()

    def test_cold_miss_cannot_commit_with_unapproved_calibration(self):
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as directory:
                router, cache, agent = fixture(directory)
                agent.temperature = [2.0, 2.0, 2.0]
                manifest = Path(directory) / "replay.json"
                router.export_replay_manifest(manifest)
                cold = restarted(directory, cache, manifest)
                rebuilt = copy.copy(agent)
                rebuilt.temperature = [1.0, 1.0, 1.0]
                with patch.object(cold, "load", return_value=rebuilt), self.assertRaises(DecisionCacheError):
                    if batch:
                        cold.predict_batch([{"state": "new", "questions": QUESTIONS, "model": "english"}])
                    else:
                        cold.predict("new", QUESTIONS, model="english")
                self.assertEqual(agent.calls, 0)
                self.assertEqual(cache.cache_info()["size"], 0)
                cache.close()

    def test_concurrent_policy_mutation_cannot_redirect_an_approved_key(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            cache.fingerprint = "policy-a"
            router.predict("hello", QUESTIONS, model="english")
            cache.fingerprint = "policy-b"
            original = agent.system_one
            def different(*args, **kwargs):
                result = original(*args, **kwargs)
                result["answers"]["keep"]["noul"] = 0.49
                return result
            agent.system_one = different
            router.predict("hello", QUESTIONS, model="english")
            cache.fingerprint = "policy-a"
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            lookup = cache._lookup_replay
            def change_namespace(ctx, **kwargs):
                cache.fingerprint = "policy-b"
                return lookup(ctx, **kwargs)
            with patch.object(cache, "_lookup_replay", side_effect=change_namespace):
                result = cold.try_replay("hello", QUESTIONS, model="english")
            self.assertEqual(result["answers"]["keep"]["noul"], 0.51)
            cache.close()

    def test_relative_source_resolution_is_bound_to_the_approved_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            previous = os.getcwd()
            try:
                os.chdir(directory)
                router.models["english"] = "checkpoint"
                router.predict("hello", QUESTIONS, model="english")
                manifest = Path(directory) / "replay.json"
                router.export_replay_manifest(manifest)
                new_directory = Path(directory) / "other-cwd"
                (new_directory / "checkpoint").mkdir(parents=True)
                os.chdir(new_directory)
                cold = Router(models={"english": "checkpoint"}, hooks=[cache],
                              replay_manifest=str(manifest), performance_mode="default")
                with self.assertRaises(DecisionCacheError):
                    cold.try_replay("hello", QUESTIONS, model="english")
            finally:
                os.chdir(previous)
                cache.close()

    def test_explicit_policy_retains_unaffected_legacy_key_layout(self):
        from laya.consistency import _canonical, _KEY_VERSION
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory)
            cache.fingerprint = "legacy-policy"
            base = hashlib.blake2b(_canonical([_KEY_VERSION, "legacy-policy", "english", None,
                                              None, None, QUESTIONS]), digest_size=16)
            base.update(b"\x00")
            base.update(_canonical("hello"))
            result = {"answers": {"keep": {"type": "noul", "noul": 0.51}}, "usage": {}}
            cache._store.add([(base.digest(), cache._encode(result), float("inf"))], 0)
            self.assertEqual(router.predict("hello", QUESTIONS, model="english")["answers"], result["answers"])
            self.assertEqual(agent.calls, 0)
            cache.close()

    def test_preparation_cli_uses_server_configuration_and_restores_environment(self):
        from laya.replay import main
        with tempfile.TemporaryDirectory() as directory:
            router, cache, _ = fixture(directory)
            router._laya_decision_cache = cache
            manifest = Path(directory) / "replay.json"
            with patch.dict(os.environ, {"LAYA_PRELOAD": "1", "LAYA_REPLAY_MANIFEST": "stale.json"}):
                def build():
                    self.assertEqual(os.environ["LAYA_PRELOAD"], "0")
                    self.assertNotIn("LAYA_REPLAY_MANIFEST", os.environ)
                    return router
                with patch("laya.serve.build_router", side_effect=build):
                    self.assertEqual(main(["--output", str(manifest), "--models", "english"]), 0)
                self.assertEqual(os.environ["LAYA_PRELOAD"], "1")
                self.assertEqual(os.environ["LAYA_REPLAY_MANIFEST"], "stale.json")
            self.assertIsNotNone(load_manifest(manifest))
            self.assertIsNone(router._laya_decision_cache)

    def test_hidden_explicit_language_has_distinct_single_batch_and_cold_keys(self):
        from laya.consistency import _canonical, _KEY_VERSION, _fingerprint
        with tempfile.TemporaryDirectory() as directory:
            router, cache, agent = fixture(directory, lang_temperatures={"de": {"temperature": [1, 1, 1]}})
            original = agent.system_one
            def language_answer(state, questions, lang=None, **kwargs):
                result = original(state, questions, **kwargs)
                result["answers"]["keep"]["noul"] = 0.49 if lang == "en" else 0.51
                return result
            agent.system_one = language_answer
            agent.predict_batch = lambda states, questions, **kwargs: [language_answer(s, questions, **kwargs) for s in states]
            # An old ambiguous row must not be used for either explicit calibration language.
            route = router.route("hello", QUESTIONS, model="english")
            base = hashlib.blake2b(_canonical([_KEY_VERSION, _fingerprint(agent), "english", route,
                                              None, None, QUESTIONS]), digest_size=16)
            base.update(b"\x00")
            base.update(_canonical("hello"))
            cache._store.add([(base.digest(), cache._encode({"answers": {"keep": {"type": "noul", "noul": 0.99}}}),
                               float("inf"))], 0)
            for language, expected in (("de", 0.51), ("en", 0.49), ("de", 0.51)):
                self.assertEqual(router.predict("hello", QUESTIONS, model="english", lang=language)["answers"]["keep"]["noul"], expected)
            self.assertEqual(agent.calls, 2)
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            cold = restarted(directory, cache, manifest)
            with patch.object(cold, "load", side_effect=AssertionError("model loaded")):
                batch = cold.predict_batch([{"state": "hello", "questions": QUESTIONS, "model": "english", "lang": code}
                                            for code in ("en", "de")])
            self.assertEqual([r["answers"]["keep"]["noul"] for r in batch], [0.49, 0.51])
            cache.close()


@unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX manifest preparation extra is not installed")
class ONNXArtifactTests(unittest.TestCase):
    def test_external_sparse_nested_and_function_tensor_files_are_all_bound(self):
        import onnx
        from onnx import helper
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            def external(name, filename, dtype=onnx.TensorProto.FLOAT):
                (directory / filename).write_bytes(b"\x00" * 8)
                tensor = onnx.TensorProto(name=name, data_type=dtype, dims=[1], data_location=onnx.TensorProto.EXTERNAL)
                tensor.external_data.add(key="location", value=filename)
                return tensor
            dense = external("dense", "dense.data")
            sparse = onnx.SparseTensorProto(values=external("values", "values.data"),
                                            indices=external("indices", "indices.data", onnx.TensorProto.INT64), dims=[1])
            nested = helper.make_graph([], "nested", [], [], [external("nested", "nested.data")])
            node = helper.make_node("If", [], [], body=nested)
            graph = helper.make_graph([node], "main", [], [], [dense])
            graph.sparse_initializer.extend([sparse])
            function = helper.make_function("local", "test", [], [],
                [helper.make_node("Constant", [], [], value=external("function", "function.data"))], [])
            model = helper.make_model(graph, functions=[function])
            path = directory / "model.onnx"
            onnx.save_model(model, str(path))
            files = _onnx_artifacts(path)
            self.assertEqual({Path(name).name for name in files},
                             {"model.onnx", "dense.data", "values.data", "indices.data", "nested.data", "function.data"})
            # Unsafe references are refused even when their target happens to exist.
            model.graph.initializer[0].external_data[0].value = "../outside.data"
            onnx.save_model(model, str(path))
            with self.assertRaises(ValueError):
                _onnx_artifacts(path)

    def test_cold_native_replay_refuses_changed_external_weights(self):
        import onnx
        from onnx import helper
        from laya.onnx_agent import _NativeONNXAgent
        with tempfile.TemporaryDirectory() as directory:
            router, cache, original = fixture(directory)
            graph = Path(directory) / "model.onnx"
            weights = Path(directory) / "weights.data"
            weights.write_bytes(b"\x00" * 4)
            tensor = onnx.TensorProto(name="w", data_type=onnx.TensorProto.FLOAT, dims=[1],
                                     data_location=onnx.TensorProto.EXTERNAL)
            tensor.external_data.add(key="location", value="weights.data")
            onnx.save_model(helper.make_model(helper.make_graph([], "fixture", [], [], [tensor])), str(graph))
            native = object.__new__(_NativeONNXAgent)
            native.__dict__.update(original.__dict__)
            native._checkpoint_signature = _checkpoint_signature(native.model_id, str(graph), {})
            router.backend = "onnx"
            router.onnx_paths = {"english": str(graph)}
            router._agents["english"] = native
            router.predict("hello", QUESTIONS, model="english")
            manifest = Path(directory) / "replay.json"
            router.export_replay_manifest(manifest)
            self.assertIn(os.path.abspath(weights), load_manifest(manifest)["models"]["english"]["files"])
            weights.write_bytes(b"changed weights")
            cold = restarted(directory, cache, manifest, backend="onnx", onnx_paths={"english": str(graph)})
            with self.assertRaises(DecisionCacheError):
                cold.try_replay("hello", QUESTIONS, model="english")
            cache.close()


class StoreIdentityTests(unittest.TestCase):
    def test_cache_context_manager_releases_sqlite_and_preserves_prediction_errors(self):
        from laya.consistency import _MemoryStore
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.sqlite"
            cache = DecisionCache(path, required=True)
            with cache as entered:
                self.assertIs(entered, cache)
                self.assertIsInstance(entered.store_id, str)
            with self.assertRaises(sqlite3.ProgrammingError):
                cache._store._db.execute("SELECT 1")
            path.rename(path.with_suffix(".closed"))
        class Store(_MemoryStore):
            closed = 0
            def close(self):
                self.closed += 1
        class Failing:
            model_id = "fixture"
            cfg = {}
            def system_one(self, *_args, **_kwargs):
                raise RuntimeError("prediction failed")
        store = Store(None)
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            with DecisionCache(store=store, required=True) as cache:
                router = Router(hooks=[cache], performance_mode="default")
                router.attach("english", Failing())
                router.predict("hello", QUESTIONS, model="english")
        self.assertEqual(store.closed, 1)

    def test_expected_identity_survives_restart_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.sqlite"
            cache = DecisionCache(path, required=True)
            identity = cache.store_id
            cache.cache_clear()
            cache.close()
            cache = DecisionCache(path, required=True, expected_store_id=identity)
            self.assertEqual(cache.store_id, identity)
            cache.close()

    def test_missing_empty_unrelated_and_replaced_stores_fail_without_initializing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.sqlite"
            with self.assertRaises(DecisionCacheError):
                DecisionCache(path, require_existing=True)
            self.assertFalse(path.exists())
            path.touch()
            before = path.read_bytes()
            with self.assertRaises(DecisionCacheError):
                DecisionCache(path, require_existing=True)
            self.assertEqual(path.read_bytes(), before)
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE unrelated (id INTEGER)")
            connection.close()
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(DecisionCacheError):
                DecisionCache(path, require_existing=True)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            path.unlink()
            first = DecisionCache(path)
            identity = first.store_id
            first.close()
            path.unlink()
            replacement = DecisionCache(path)
            replacement.close()
            with self.assertRaises(DecisionCacheError):
                DecisionCache(path, expected_store_id=identity)

    def test_legacy_store_identity_can_be_explicitly_adopted_but_not_forged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE laya_decisions (key BLOB PRIMARY KEY, touched_at REAL NOT NULL, "
                                   "expires_at REAL NOT NULL, value BLOB NOT NULL) WITHOUT ROWID")
            connection.close()
            cache = DecisionCache(path, require_existing=True)
            self.assertIsInstance(cache.store_id, str)
            cache.close()
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE laya_cache_metadata SET value='corrupt' WHERE name='store_id'")
            connection.close()
            with self.assertRaises(DecisionCacheError):
                DecisionCache(path, required=True)

    def test_invalid_guard_arguments_and_nondurable_stores_are_refused(self):
        for kwargs in ({"require_existing": 1}, {"expected_store_id": "invalid"},
                       {"require_existing": True}, {"expected_store_id": "00000000-0000-0000-0000-000000000000"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                DecisionCache(**kwargs)
        with self.assertRaises(ValueError):
            Router(hooks=[DecisionCache()]).export_replay_manifest("unused.json")


if __name__ == "__main__":
    unittest.main()
