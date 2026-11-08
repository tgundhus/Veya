"""Parsed tokenizer reuse must track vocabulary changes and keep a bounded working set."""
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya import agent as runtime  # noqa: E402


class TokenizerParseCacheTests(unittest.TestCase):
    def setUp(self):
        runtime._TOKENIZERS.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.loaded = []

    def tearDown(self):
        runtime._TOKENIZERS.clear()
        self.temp.cleanup()

    def directory(self, name):
        directory = self.root / name
        directory.mkdir()
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (directory / "tokenizer.json").write_text("vocabulary one", encoding="utf-8")
        return str(directory)

    def load(self, directory):
        tokenizer = object()
        self.loaded.append((directory, tokenizer))
        return tokenizer

    def test_vocabulary_change_invalidates_without_a_config_change(self):
        directory = self.directory("checkpoint")
        config = Path(directory) / "tokenizer_config.json"
        config_stamp = config.stat().st_mtime_ns
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=self.load):
            first = runtime._load_tokenizer(directory, {})
            self.assertIs(runtime._load_tokenizer(directory, {}), first)
            (Path(directory) / "tokenizer.json").write_text("a different vocabulary", encoding="utf-8")
            second = runtime._load_tokenizer(directory, {})
        self.assertEqual(config.stat().st_mtime_ns, config_stamp)
        self.assertIsNot(second, first)
        self.assertEqual(len(self.loaded), 2)
        self.assertEqual(len(runtime._TOKENIZERS), 1)

    def test_bounded_cache_reuses_recent_entries(self):
        directories = [self.directory("checkpoint-%d" % index) for index in range(runtime._TOKENIZERS_MAX + 1)]
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=self.load):
            held = [runtime._load_tokenizer(directory, {}) for directory in directories[:-1]]
            self.assertIs(runtime._load_tokenizer(directories[0], {}), held[0])
            runtime._load_tokenizer(directories[-1], {})
            self.assertEqual(len(runtime._TOKENIZERS), runtime._TOKENIZERS_MAX)
            self.assertIs(runtime._load_tokenizer(directories[0], {}), held[0])
            self.assertIsNot(runtime._load_tokenizer(directories[1], {}), held[1])

    def test_concurrent_loads_share_one_parse(self):
        directory = self.directory("shared")
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=self.load):
            with ThreadPoolExecutor(max_workers=8) as workers:
                tokenizers = list(workers.map(lambda _: runtime._load_tokenizer(directory, {}), range(16)))
        self.assertEqual(len(self.loaded), 1)
        self.assertTrue(all(tokenizer is tokenizers[0] for tokenizer in tokenizers))


if __name__ == "__main__":
    unittest.main()
