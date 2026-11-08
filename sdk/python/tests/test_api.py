import importlib.metadata
import inspect
from pathlib import Path
import sys
import unittest

import veya_client
from veya_client import VeyaClient, VeyaConnectionError, VeyaError, VeyaTimeoutError


class APITests(unittest.TestCase):
    def test_legacy_imports_share_the_same_classes(self):
        import laya_client
        from laya_client.client import LayaClient
        self.assertIs(laya_client, veya_client)
        self.assertIs(LayaClient, VeyaClient)
        self.assertIs(laya_client.LayaAPIError, veya_client.VeyaAPIError)

    def test_public_exports(self):
        self.assertEqual(set(veya_client.__all__), {
            "VeyaClient", "VeyaError", "VeyaValidationError", "VeyaAPIError",
            "VeyaConnectionError", "VeyaTimeoutError", "VeyaResponseError",
        })
        for name in veya_client.__all__:
            if name.endswith("Error"):
                self.assertTrue(issubclass(getattr(veya_client, name), VeyaError))
        self.assertTrue(issubclass(VeyaTimeoutError, VeyaConnectionError))

    def test_predict_parameters(self):
        parameters = inspect.signature(VeyaClient.predict).parameters
        self.assertEqual(list(parameters), ["self", "state", "questions", "model", "task", "lang",
                                           "lang_guess", "max_len", "head_max_len", "min_confidence", "timeout"])
        for name in list(parameters)[3:]:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)
        self.assertEqual(list(inspect.signature(VeyaClient.health).parameters), ["self", "timeout"])

    def test_predict_batch_parameters(self):
        from veya_client.types import BatchPrediction
        signature = inspect.signature(VeyaClient.predict_batch)
        parameters = signature.parameters
        self.assertEqual(list(parameters), ["self", "states", "questions", "model", "task", "lang",
                                           "lang_guess", "max_len", "head_max_len", "min_confidence",
                                           "batch_size", "sort_by_length", "timeout"])
        for name in list(parameters)[3:]:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)
        self.assertIs(signature.return_annotation, BatchPrediction)
        self.assertEqual(BatchPrediction.__required_keys__, {"results", "total_usage"})

    def test_constructor_defaults(self):
        parameters = inspect.signature(VeyaClient).parameters
        self.assertEqual({k: p.default for k, p in parameters.items()},
                         {"base_url": "http://127.0.0.1:8000", "api_key": None, "timeout": 120.0})
        self.assertTrue(all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in parameters.values()))

    def test_types_match_strict_response_requirements(self):
        from veya_client.types import ChoiceAnswer, NoulAnswer, Prediction, ScoreAnswer
        self.assertIn("confidence", ChoiceAnswer.__required_keys__)
        self.assertIn("confidence", ScoreAnswer.__required_keys__)
        self.assertIn("confidence", NoulAnswer.__optional_keys__)
        self.assertIn("abstention", ChoiceAnswer.__optional_keys__)
        self.assertIn("routing", Prediction.__optional_keys__)
        for answer in (ChoiceAnswer, NoulAnswer, ScoreAnswer):
            self.assertIn("decision_policy", answer.__optional_keys__)
            self.assertIn("confidence_calibrated", answer.__optional_keys__)

    def test_installed_package_has_no_runtime_requirements(self):
        distribution = importlib.metadata.distribution("veya-client")
        self.assertFalse(distribution.requires)
        installed = Path(distribution.locate_file("veya_client/__init__.py")).resolve()
        self.assertEqual(Path(veya_client.__file__).resolve(), installed)
        self.assertNotIn("src", installed.parts)
        self.assertTrue(Path(distribution.locate_file("veya_client/py.typed")).is_file())
        self.assertTrue(any("LICENSE" in str(p) for p in distribution.files))

    def test_no_inference_or_http_dependencies_in_client_environment(self):
        names = {d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()}
        forbidden = {"veya", "laya", "torch", "transformers", "numpy", "safetensors",
                     "huggingface-hub", "requests", "httpx"}
        self.assertFalse(names & forbidden, names & forbidden)
        self.assertFalse(set(sys.modules) & {"veya", "torch", "transformers", "numpy"})

    def test_typed_question_and_extension_contracts(self):
        from typing import get_args
        from veya_client.types import (ChoiceQuestion, Health, MinConfidenceKey, NoulAnswer,
                                       NoulQuestion, ScoreQuestion, Tournament, Usage)
        from veya_client._validation import MIN_CONFIDENCE_KEYS
        self.assertEqual(set(get_args(MinConfidenceKey)), MIN_CONFIDENCE_KEYS)
        for question in (ChoiceQuestion, ScoreQuestion):
            self.assertEqual(question.__required_keys__, {"type", "instructions", "criteria"})
            self.assertEqual(question.__optional_keys__, {"option_order"})
        self.assertEqual(NoulQuestion.__required_keys__, {"type", "instructions"})
        self.assertEqual(Tournament.__required_keys__, {"finalists", "rounds"})
        self.assertNotIn("x_tournament", NoulAnswer.__annotations__)
        self.assertEqual(Usage.__required_keys__, {"input_tokens", "output_tokens"})
        self.assertTrue({"decision_policy", "policy_probe_tokens"} <= Usage.__optional_keys__)
        self.assertEqual(Health.__optional_keys__, {"loaded", "device", "model_free_replay"})


if __name__ == "__main__":
    unittest.main()
