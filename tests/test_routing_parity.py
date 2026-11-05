"""Versioned cross-SDK routing contract, independently checked against recorded fixtures."""
import json
import importlib.util
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya import lang  # noqa: E402
from laya import _routing_unicode as unicode_rules  # noqa: E402
from research.scripts.generate_routing_corpus import PROSE, expand, expected  # noqa: E402


CORPUS = json.loads((ROOT / "tests/fixtures/routing-parity-v1.json").read_text(encoding="utf-8"))


class RoutingContractTests(unittest.TestCase):
    def test_language_vocabulary_cannot_drift_between_ports(self):
        source = (ROOT / "laya-ts/src/lang.ts").read_text(encoding="utf-8")
        source = re.sub(r"//[^\n]*", "", source)

        def strings(body):
            return {json.loads(literal) for literal in re.findall(r'"(?:[^"\\]|\\.)*"', body)}

        stop = source.split("const STOP:", 1)[1].split("const NON_EN_DIACRITICS", 1)[0]
        found = {name: strings(body) for name, body in re.findall(r"(\w+): new Set\(\[(.*?)\]\)", stop, re.S)}
        self.assertEqual(found, lang._STOP)
        diacritics = source.split("const NON_EN_DIACRITICS =", 1)[1].split("export const", 1)[0]
        self.assertEqual(set("".join(strings(diacritics))), lang._NON_EN_DIACRITICS)

    def test_recorded_cross_sdk_contract(self):
        self.assertEqual(lang.ROUTING_CONTRACT_VERSION, CORPUS["contract"])
        self.assertEqual(unicode_rules.VERSION, CORPUS["unicode_version"])
        self.assertEqual(unicode_rules.TABLE_SHA256, CORPUS["unicode_table_sha256"])
        for row in CORPUS["cases"]:
            with self.subTest(case=row["id"]):
                self.assertEqual(expected(row), row["expected"])

    def test_explanation_does_not_change_the_verdict(self):
        for row in CORPUS["cases"]:
            with self.subTest(case=row["id"]):
                state = expand(row["state"])
                explained = lang.analyse(state, explain=True)
                self.assertIn("diagnostic", explained)
                explained.pop("diagnostic")
                self.assertEqual(explained, lang.analyse(state))

    def test_numeric_state_keys_follow_ecmascript_object_order(self):
        state = {"10": "ten", "02": "leading", "2": "two", "0": "zero", "4294967295": "tail"}
        self.assertEqual(lang.state_text(state), "zero two ten leading tail")
        self.assertEqual(list(state), ["10", "02", "2", "0", "4294967295"])
        self.assertEqual(lang.state_text({10: "ten", 2: "two"}), "ten two")

    def test_enumerating_keys_does_not_read_unvisited_values(self):
        from collections.abc import Mapping

        class Bounded(Mapping):
            def __iter__(self):
                return iter(("first", "later"))

            def __len__(self):
                return 2

            def __getitem__(self, key):
                if key == "first":
                    return "hello"
                raise AssertionError("unvisited value read")

        self.assertEqual(lang.state_text(Bounded(), 2), "he")

    def test_pinned_unicode_survives_a_different_host_database(self):
        native = unicode_rules._NATIVE_VERSION
        unicode_rules._NATIVE_VERSION = False
        try:
            self.assertEqual(unicode_rules.unicode_lower("AΣ AΣ\u0345 AΣA AΣ'"), "aς aς\u0345 aσa aς'")
            self.assertEqual(unicode_rules.unicode_lower("\ua7cb \u1c89 İ"), "\ua7cb \u1c89 i\u0307")
            self.assertEqual(unicode_rules.unicode_repr("\U0001e030"), "'\U0001e030'")
            self.assertEqual(unicode_rules.unicode_repr("\u1c89"), "'\\u1c89'")
            self.assertEqual(unicode_rules.unicode_repr("a'\n\t\\\ud800"), '"a\'\\n\\t\\\\\\ud800"')
            for row in CORPUS["cases"]:
                with self.subTest(case=row["id"]):
                    self.assertEqual(expected(row), row["expected"])
        finally:
            unicode_rules._NATIVE_VERSION = native

    def test_invalid_bom_model_alias_is_not_silently_trimmed(self):
        from laya.router import normalise_name
        with self.assertRaises(ValueError):
            normalise_name("\ufeffen")

    def test_entire_analysis_uses_frozen_rules_on_other_python_versions(self):
        spec = importlib.util.spec_from_file_location("laya._frozen_lang_probe", ROOT / "laya/lang.py")
        frozen = importlib.util.module_from_spec(spec)
        with patch("unicodedata.unidata_version", "0.0.0"), patch.object(unicode_rules, "_NATIVE_VERSION", False):
            spec.loader.exec_module(frozen)
            self.assertFalse(frozen._NATIVE_UNICODE)
            for row in CORPUS["cases"]:
                with self.subTest(case=row["id"]):
                    state = expand(row["state"])
                    self.assertEqual(frozen.analyse(state, explain=True), lang.analyse(state, explain=True))
                    self.assertEqual(frozen.latin_profile(frozen.state_text(state)), lang.latin_profile(lang.state_text(state)))

    def test_fixed_seed_live_cross_runtime_parity(self):
        node = shutil.which("node")
        compiler = ROOT / "laya-ts/node_modules/typescript/bin/tsc"
        if not node or not compiler.is_file():
            self.skipTest("Node and the local TypeScript dev toolchain are needed for live cross-runtime checks")
        subprocess.run([node, str(compiler), "-p", "tsconfig.json"], cwd=ROOT / "laya-ts", check=True,
                       capture_output=True, encoding="utf-8", timeout=90)
        rng = random.Random(231105)
        words = ["where", "why", "done", "already", "the", "and", "i", "please", "je", "nous", "pour",
                 "merci", "ich", "mir", "bitte", "heute", "para", "quero", "agora", "com", "come", "do",
                 "per", "da", "un", "la", "e", "İ", "Ⅳ", "ᴬ", "\u0345", "😀", "α/com", "α(", "\u1c89"]
        rows = list(CORPUS["cases"])
        for language, words_in_language in lang._STOP.items():
            for word in sorted(words_in_language):
                rows.append({"id": "vocabulary-%s-%s" % (language, word), "state": " ".join([word] * 4)})
        for index in range(3000):
            text = " ".join(rng.choices(words, k=rng.randint(0, 30)))
            if index % 3 == 0:
                state = {"note": "Please review the invoice and tell me what happened. " * rng.randrange(0, 90),
                         "customer": text + "\n" + rng.choice(list(PROSE.values()))}
            elif index % 3 == 1:
                state = [text, {"10": text, "2": rng.choice(list(PROSE.values()))}]
            else:
                state = text
            rows.append({"id": "seeded-%d" % index, "state": state})
        result = subprocess.run([node, str(ROOT / "tests/routing_parity_runner.mjs")], cwd=ROOT, check=True,
                                input=json.dumps(rows, ensure_ascii=True), capture_output=True,
                                encoding="utf-8", timeout=90)
        actual = json.loads(result.stdout)
        self.assertEqual(len(actual), len(rows))
        for row, observed in zip(rows, actual):
            with self.subTest(case=row["id"]):
                self.assertEqual(observed, expected(row))


if __name__ == "__main__":
    unittest.main()
