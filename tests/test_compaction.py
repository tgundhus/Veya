"""Run with python tests/test_compaction.py; no checkpoint or network required."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya import ContextCompactor, Router  # noqa: E402
from veya.confidence import apply_confidence_gate  # noqa: E402


def transcript(count=3):
    messages = [{"role": "system", "text": "Never change generated files."}]
    for i in range(count):
        messages.extend([
            {"role": "assistant", "text": "", "toolUses": [
                {"tool_use_id": str(i), "tool": "Read", "input": {"file_path": "file%d" % i},
                 "replayable": True}]},
            {"role": "user", "text": "", "toolResults": [
                {"tool_use_id": str(i), "text": "snapshot content " * 150}]},
        ])
    messages.append({"role": "user", "text": "Only fix file2 now."})
    return messages


class Scorer:
    def __init__(self, scores=(0.01, 0.01), usage=None):
        self.scores = scores
        self.usage = {"truncated": False, "state_tokens_dropped": 0} if usage is None else usage
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((copy.deepcopy(states), copy.deepcopy(questions), kwargs))
        return [{"answers": {"keep_call": {"noul": self.scores[0]},
                             "keep_result": {"noul": self.scores[1]}}, "usage": self.usage}
                for _ in states]


class CompactionTests(unittest.TestCase):
    def compact(self, messages=None, scorer=None, **kwargs):
        return ContextCompactor(scorer or Scorer(), preserve_recent_messages=0,
                                **kwargs).compact(transcript() if messages is None else messages,
                                                  goal="Fix file2", constraints="Keep generated files unchanged")

    def test_default_protection_and_explicit_replayability(self):
        messages = transcript()
        output = ContextCompactor(Scorer()).compact(messages, goal="Fix file2")
        self.assertEqual([d["action"] for d in output["decisions"]], ["keep"] * 3)
        self.assertEqual(output["messages"], messages)
        messages = transcript(1)
        del messages[1]["toolUses"][0]["replayable"]
        self.assertEqual(self.compact(messages)["decisions"][0]["reason"], "not_replayable")

    def test_drop_pair_preserves_text_metadata_and_original(self):
        messages = transcript()
        messages[1]["text"] = "Exact assistant wording."
        messages[2]["metadata"] = {"snapshot": [1, 2]}
        original = copy.deepcopy(messages)
        result = self.compact(messages)
        self.assertEqual(messages, original)
        self.assertEqual([d["action"] for d in result["decisions"]], ["drop_pair"] * 3)
        self.assertEqual(result["messages"][1]["text"], "Exact assistant wording.")
        self.assertEqual(result["messages"][2]["metadata"], {"snapshot": [1, 2]})
        result["messages"][2]["metadata"]["snapshot"].append(3)
        self.assertEqual(messages, original)

    def test_result_truncation_keeps_call_and_mirrored_output(self):
        messages = transcript(1)
        messages[1]["toolUses"][0]["text"] = messages[2]["toolResults"][0]["text"]
        result = self.compact(messages, Scorer((0.99, 0.01)), truncate_head_chars=12)
        self.assertEqual(result["decisions"][0]["action"], "truncate_result")
        use = result["messages"][1]["toolUses"][0]
        output = result["messages"][2]["toolResults"][0]
        self.assertEqual(use["text"], output["text"])
        self.assertEqual(output["text"], "snapshot con\n[veya: replayable output omitted]")

    def test_uncertainty_retains_everything(self):
        for scores in ((0.01, 0.5), (0.5, 0.5), (0.99, 0.99)):
            with self.subTest(scores=scores):
                result = self.compact(scorer=Scorer(scores))
                self.assertEqual(result["messages"], transcript())
                self.assertTrue(all(d["action"] == "keep" for d in result["decisions"]))

    def test_abstention_on_either_judgment_prevents_pruning(self):
        for key in ("keep_call", "keep_result"):
            for flags in ({"low_confidence": True}, {"abstention": "abstained"},
                          {"abstention": "unevaluated"}, {"abstention": None}, {"abstention": "unknown"}):
                with self.subTest(key=key, flags=flags):
                    class Flagged(Scorer):
                        def predict_batch(self, *args, **kwargs):
                            results = super().predict_batch(*args, **kwargs)
                            for result in results:
                                result["answers"][key].update(flags)
                            return results
                    result = self.compact(scorer=Flagged())
                    self.assertEqual(result["messages"], transcript())
                    self.assertTrue(all(d["reason"] == "unverified_confidence" for d in result["decisions"]))
                    self.assertEqual(result["stats"]["pairs_dropped"], 0)
                    self.assertEqual(result["stats"]["results_shortened"], 0)

    def test_core_confidence_gate_is_honored_and_passed_answers_can_prune(self):
        class Gated(Scorer):
            threshold = 0.999

            def predict_batch(self, *args, **kwargs):
                results = super().predict_batch(*args, **kwargs)
                for result in results:
                    for answer in result["answers"].values():
                        answer.update(type="noul", answer_confidence=0.99)
                apply_confidence_gate(results, self.threshold)
                return results
        scorer = Gated()
        self.assertEqual(self.compact(scorer=scorer)["messages"], transcript())
        scorer.threshold = 0.9
        self.assertEqual(self.compact(scorer=scorer)["stats"]["pairs_dropped"], 3)

    def test_benchmark_injection_is_in_visible_scoring_excerpt(self):
        from veya.compaction import _excerpt
        from research.scripts.bench_compaction import cases
        case = next(c for c in cases() if c["id"] == "untrusted_output")
        scorer = Scorer()
        self.compact(case["messages"], scorer)
        state = scorer.calls[0][0][0]
        self.assertIn(case["required_text"], state["output_excerpt"])
        self.assertEqual(state["output_excerpt"], _excerpt(case["messages"][2]["toolResults"][0]["text"]))

    def test_regression_suite_is_registered_in_ci_and_release_gate(self):
        root = Path(__file__).resolve().parents[1]
        suites = runpy.run_path(str(root / "scripts/test_suites.py"))
        self.assertIn("tests/test_compaction.py", suites["SCRIPT_SUITES"])

    def test_pins_errors_pending_and_multimodal_are_protected(self):
        for location, key, value in ((1, "pinned", True), (2, "pinned", True),
                                     (1, "content", [{"type": "image"}])):
            messages = transcript(1)
            messages[location][key] = value
            self.assertEqual(self.compact(messages)["decisions"][0]["reason"], "pinned")
        for key, value in (("pinned", True), ("isError", True), ("content", [])):
            messages = transcript(1)
            messages[2]["toolResults"][0][key] = value
            self.assertEqual(self.compact(messages)["messages"], messages)
        messages = transcript(1)
        del messages[2]["toolResults"]
        self.assertEqual(self.compact(messages)["decisions"][0]["reason"], "incomplete")

    def test_context_truncation_and_missing_usage_fail_closed(self):
        for usage in ({}, {"truncated": True, "state_tokens_dropped": 200},
                      {"truncated": False, "state_tokens_dropped": 1},
                      {"truncated": False, "state_tokens_dropped": 0, "truncated_questions": ["keep_call"]}):
            result = self.compact(scorer=Scorer(usage=usage))
            self.assertEqual(result["messages"], transcript())
            self.assertTrue(all(d["reason"] == "unverified_context" for d in result["decisions"]))

    def test_failure_in_later_batch_rolls_back_earlier_decisions(self):
        class Failing(Scorer):
            def predict_batch(self, *args, **kwargs):
                if self.calls:
                    raise RuntimeError("private transcript must not enter diagnostics")
                return super().predict_batch(*args, **kwargs)
        result = self.compact(scorer=Failing(), batch_size=1)
        self.assertEqual(result["messages"], transcript())
        self.assertEqual(result["stats"]["fallback"], "scoring_failed:RuntimeError")
        self.assertNotIn("private transcript", json.dumps(result))
        self.assertTrue(all(d["action"] == "keep" for d in result["decisions"]))

    def test_malformed_results_roll_back(self):
        for value in (True, -0.1, 1.1, float("nan"), None):
            result = self.compact(scorer=Scorer((value, 0.01)))
            self.assertEqual(result["messages"], transcript())
            self.assertEqual(result["stats"]["fallback"], "scoring_failed:ValueError")
        class Missing(Scorer):
            def predict_batch(self, *args, **kwargs):
                return []
        self.assertEqual(self.compact(scorer=Missing())["messages"], transcript())

    def test_token_counter_controls_acceptance(self):
        messages = transcript()
        compactor = ContextCompactor(Scorer(), preserve_recent_messages=0)
        result = compactor.compact(messages, goal="Fix file2", token_counter=lambda _: 100)
        self.assertEqual(result["messages"], messages)
        self.assertEqual(result["stats"]["fallback"], "insufficient_reduction")
        self.assertEqual(result["stats"]["tokens_before"], result["stats"]["tokens_after"])
        result = compactor.compact(messages, goal="Fix file2", token_counter=lambda m: len(json.dumps(m)))
        self.assertEqual(result["stats"]["reduction_unit"], "tokens")
        self.assertGreater(result["stats"]["reduction_ratio"], 0.1)

    def test_state_contains_goal_constraints_and_output_excerpts(self):
        scorer = Scorer()
        self.compact(scorer=scorer, batch_size=2)
        self.assertEqual([len(c[0]) for c in scorer.calls], [2, 1])
        state = scorer.calls[0][0][0]
        self.assertEqual(state["goal"], "Fix file2")
        self.assertEqual(state["constraints"], "Keep generated files unchanged")
        self.assertIn("Never change generated files.", state["instructions"])
        self.assertEqual(state["recent"][-1]["text"], "Only fix file2 now.")
        self.assertIn("[output middle omitted]", state["output_excerpt"])

    def test_shared_messages_and_mixed_decisions_keep_pair_integrity(self):
        messages = transcript(2)
        # Two parallel calls share a message; one result will be kept.
        messages[1]["toolUses"] += messages[3].pop("toolUses")
        class Mixed(Scorer):
            def predict_batch(self, states, questions, **kwargs):
                return [Scorer((0.01, 0.01) if s["input"]["file_path"] == "file0" else (0.99, 0.99))
                        .predict_batch([s], questions)[0] for s in states]
        result = self.compact(messages, Mixed())
        self.assertEqual([d["action"] for d in result["decisions"]], ["drop_pair", "keep"])
        uses = [b["tool_use_id"] for m in result["messages"] for b in m.get("toolUses", [])]
        results = [b["tool_use_id"] for m in result["messages"] for b in m.get("toolResults", [])]
        self.assertEqual(uses, ["1"])
        self.assertEqual(results, uses)
        # An originally empty message stays intact, even when other messages are pruned.
        self.assertIn({"role": "assistant", "text": ""}, result["messages"])
        self.assertEqual(result["stats"]["pairs_dropped"], 1)

    def test_predictor_cannot_mutate_input_or_later_batch_context(self):
        class Mutating(Scorer):
            def predict_batch(self, states, questions, **kwargs):
                result = super().predict_batch(states, questions, **kwargs)
                states[0]["input"]["file_path"] = "changed"
                states[0]["instructions"].clear()
                states[0]["recent"].clear()
                return result
        messages = transcript()
        original = copy.deepcopy(messages)
        scorer = Mutating()
        self.compact(messages, scorer, batch_size=1)
        self.assertEqual(messages, original)
        self.assertTrue(all(c[0][0]["instructions"] for c in scorer.calls))
        self.assertTrue(all(c[0][0]["recent"] for c in scorer.calls))

    def test_short_results_never_expand_and_reduction_gate_rolls_back(self):
        messages = transcript(1)
        messages[2]["toolResults"][0]["text"] = "short result"
        result = self.compact(messages, Scorer((0.99, 0.01)), min_reduction_ratio=0)
        self.assertEqual(result["messages"], messages)
        self.assertEqual(result["decisions"][0]["action"], "keep")
        self.assertEqual(result["stats"]["fallback"], "insufficient_reduction")

    def test_router_request_shape_and_model_budget_forwarding(self):
        class LocalRouter(Router):
            def predict_batch(self, requests, **kwargs):
                self.requests = requests
                self.kwargs = kwargs
                return Scorer().predict_batch([r["state"] for r in requests], requests[0]["questions"])
        router = LocalRouter()
        compactor = ContextCompactor(router, model="english", max_len=1024, preserve_recent_messages=0)
        result = compactor.compact(transcript(), goal="Fix file2")
        self.assertIsNone(result["stats"]["fallback"])
        self.assertTrue(all(r["model"] == "english" and r["max_len"] == 1024 for r in router.requests))
        self.assertEqual(router.kwargs["batch_size"], 8)

    def test_empty_and_no_candidate_transcripts_do_not_call_model(self):
        class Unexpected(Scorer):
            def predict_batch(self, *args, **kwargs):
                raise AssertionError("should never be called")
        for messages in ([], [{"role": "user", "text": ""}], [{"role": "user", "text": "hello"}]):
            result = self.compact(messages, Unexpected())
            self.assertEqual(result["messages"], messages)
            self.assertEqual(result["stats"]["batches"], 0)

    def test_invalid_inputs_are_rejected_before_inference(self):
        for kwargs in ({"batch_size": True}, {"drop_threshold": 0.6}, {"drop_threshold": float("nan")},
                       {"truncate_head_chars": -1}, {"max_len": 0}, {"min_reduction_ratio": 2}):
            with self.assertRaises(ValueError):
                ContextCompactor(Scorer(), **kwargs)
        for mutate in (lambda m: m[2]["toolResults"].append(m[2]["toolResults"][0]),
                       lambda m: m[2]["toolResults"][0].update(tool_use_id="unknown"),
                       lambda m: m[1]["toolUses"][0].update(replayable="true"),
                       lambda m: m[1]["toolUses"][0]["input"].update(bad=float("nan"))):
            messages = transcript()
            mutate(messages)
            with self.assertRaises(ValueError):
                self.compact(messages)


if __name__ == "__main__":
    unittest.main()
