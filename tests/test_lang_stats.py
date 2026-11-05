"""`detect_script` / `script_profile` must be unchanged by the single-pass refactor in #lang.

The reference implementations below are the pre-refactor versions; the new ones share one pass
over the text, so these checks pin that the counts, the fractions and the tie-break are the same.

Run: python tests/test_lang_stats.py
"""
import os
import random
import sys
from collections import UserDict
from types import MappingProxyType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.lang import _SCRIPT_RANGES, _iter_text, analyse, detect_script, script_profile, state_text  # noqa: E402
from research.scripts.bench_state_text import eager_state_text, eager_text_leaves  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def _is_latin_cp(cp):
    return cp < 0x02B0 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A


def ref_counts(text):
    counts = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if _is_latin_cp(cp):
            latin += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    counts["latin"] = latin
    return counts


def ref_detect(text):
    counts = ref_counts(text)
    if sum(counts.values()) == 0:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def ref_profile(text):
    counts = {"latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if _is_latin_cp(cp):
            counts["latin"] += 1
            continue
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items() if v}


TEXTS = [
    "",
    "   ",
    "hello world",
    "hello мир",
    "мир",
    "世界",
    "日本語です",
    "ΑΒΓΔ",
    "abc ᚠᚢᚦ",           # an unlisted script lands under "other"
    "ab αβ",              # latin/greek tie
    "αβ аб",              # greek/cyrillic tie
    "ᚠᚢᚦabc",            # other/latin tie
    "Hallo, meine Bestellung ist zweimal abgebucht worden.",
    "مرحبا كيف حالك",
    "İstanbul",
]

for text in TEXTS:
    check("detect_script(%r)" % text, detect_script(text), ref_detect(text))
    check("script_profile(%r)" % text, script_profile(text), ref_profile(text))

# analyse uses the same counts, so its profile must match too, and it must not regress on a
# non-Latin override (a lowercase non-Latin run flips the dominant script).
for text in ("hello мир", "мир", "مرحبا كيف حالك", "hello world"):
    check("analyse.profile(%r)" % text, analyse(text)["script_profile"], ref_profile(text))
check("analyse/arabic", analyse("مرحبا كيف حالك")["script"], "arabic")
check("analyse/empty is unknown", analyse("")["script"], "unknown")

# a per-script fraction vector still sums to 1
for text in ("hello мир", "abc ᚠᚢᚦ"):
    total = sum(script_profile(text).values())
    check("profile sums to 1 (%r)" % text, round(total, 9), 1.0)


# Bounded state flattening retains the eager implementation's outputs and diagnostic paths.
STATES = [None, 123, "", "hello мир 🌍", b"hello", bytearray("κόσμος", "utf-8"), b"\xff",
          ["", "", "last"], ("hello", {"message": "你好", "ignored_number": 3}),
          UserDict({"body": "Bonjour"}), MappingProxyType({"body": "Hallo"}),
          {"x" * 129: {"body": "wide path"}, 2 ** 100: "wide integer", True: "boolean key"}]
deep = "too deep"
for _ in range(8):
    deep = [deep]
STATES.append(deep)
cycle = ["cycle text"]
cycle.append(cycle)
STATES.append(cycle)
rng = random.Random(0)
for _ in range(100):
    STATES.append({str(i): rng.choice([None, 3, "", "hello 🌍", b"\xff", ["one", "two"]])
                   for i in range(rng.randrange(12))})
for index, state in enumerate(STATES):
    for budget in (-2, -1, 0, 1, 2, 5, 4000):
        check("state_text/%d/budget=%d" % (index, budget),
              state_text(state, budget), eager_state_text(state, budget))
    paths, ref_paths = [], []
    check("state_text/%d/leaves" % index, list(_iter_text(state, _paths=paths)),
          eager_text_leaves(state, paths=ref_paths))
    check("state_text/%d/paths" % index, paths, ref_paths)


class BudgetGuard(list):
    def __iter__(self):
        yield "full"
        raise AssertionError("flattening walked past its filled budget")


check("state_text/stops when budget is filled", state_text(BudgetGuard(), 4), "full")


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all lang stats tests passed")
sys.exit(1 if FAIL else 0)
