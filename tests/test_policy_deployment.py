"""The optional inference-policy override must survive the shipped deployments."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PolicyDeploymentTests(unittest.TestCase):
    def test_http_compose_passes_the_policy_without_selecting_a_default(self):
        source = (ROOT / "compose.http.yaml").read_text(encoding="utf-8")
        values = re.findall(r'^\s+VEYA_DECISION_POLICY: "(.*)"$', source, re.MULTILINE)
        self.assertEqual(values, ["${VEYA_DECISION_POLICY:-}"])

    def test_nix_policy_is_optional_and_defers_validation_to_the_runtime(self):
        source = (ROOT / "nix/veya-serve.nix").read_text(encoding="utf-8")
        option = re.search(r"decisionPolicy = lib\.mkOption \{(.*?)\n    \};", source, re.DOTALL)
        self.assertIsNotNone(option)
        self.assertIn("type = lib.types.nullOr lib.types.str;", option[1])
        self.assertRegex(option[1], r"default = null;")
        self.assertRegex(source, r"lib\.optionalAttrs \(cfg\.decisionPolicy != null\) \{\s*"
                                r"VEYA_DECISION_POLICY = cfg\.decisionPolicy;")


if __name__ == "__main__":
    unittest.main()
