"""Language detection's shortcuts must return what the plain per-character code returns.

`_script_counts` counts ASCII text in C, `latin_profile` scores each distinct word once and skips
an identifier scan that cannot match, and `_named_prose_language` searches a line once before
searching its tokens. The reference functions below are the code those replaced, kept verbatim,
and seeded random text in many scripts must give identical results through both.

Run: python tests/test_lang_fast_paths.py
"""
import os
import random
from collections import Counter
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veya import lang  # noqa: E402
from veya.lang import (  # noqa: E402
    _CODE_LINE, _EN_COLLISION_WORDS, _EN_ONLY_WORDS, _IDENTIFIER, _JOINED, _LETTER_RUN,
    _NON_EN_DIACRITICS, _NORDIC_OVERLAP_WORDS, _SCRIPT_RANGES, _SHARED_WORDS, _SHORT_SWEDISH_WORDS,
    _STOP, _WORD, NON_EN_DIACRITIC_RATE, _english_rescued_by_words, _shouted, _shouted_evidence,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def ref_script_counts(text):
    counts = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x02B0 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
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


# The references are the per-character code of the original project's release that Veya merged
# (0.3.29): the same rules (Nordic overlap, short Swedish phrases, collision words counted once,
# shouted lines), without the shortcuts. The shortcuts must give exactly what these give.
def ref_latin_profile(text):
    words = _WORD.findall(_IDENTIFIER.sub(" ", text).replace("İ", "i").lower())
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))
    non_english = diac_rate >= NON_EN_DIACRITIC_RATE
    nordic_overlap = bool(set(words) & _NORDIC_OVERLAP_WORDS) and not bool(set(words) & _EN_ONLY_WORDS)
    if 1 < len(words) < 4 and set(words) & _SHORT_SWEDISH_WORDS:
        return {"language": "sv", "english_hits": 0, "diacritic_rate": diac_rate,
                "looks_non_english": non_english}
    if len(words) < 4:
        return {"language": None, "english_hits": 0, "diacritic_rate": diac_rate,
                "looks_non_english": non_english or nordic_overlap}
    counts = Counter(words)
    scores = {lg: sum(1 if w in _EN_COLLISION_WORDS else n for w, n in counts.items() if w in sw)
              for lg, sw in _STOP.items()}
    en = scores.get("en", 0)
    evidenced = {lg: s for lg, s in scores.items()
                 if lg != "en" and any(w not in _SHARED_WORDS for w in set(words) & _STOP[lg])}
    best_lg, best = max(evidenced.items(), key=lambda kv: kv[1], default=(None, 0))
    language = None
    if best_lg and best >= max(2, en + 2):
        language = best_lg
    elif (best_lg == "sv" and "inte" in words and "kan" in words
          and words[0] in {"kan", "jag", "vi"} and en <= 1):
        language = best_lg
    elif best_lg and non_english and best >= max(2, en):
        language = best_lg
    elif en and (not non_english or _english_rescued_by_words(words, diac_rate)):
        language = "en"
    return {"language": language, "english_hits": en, "diacritic_rate": diac_rate,
            "looks_non_english": non_english or (language is None and nordic_overlap)}


def ref_named_prose_language(segment):
    if not segment.strip() or _CODE_LINE.search(segment):
        return None
    prose = " ".join(tok for tok in segment.split() if not _JOINED.search(tok))
    shouted = _shouted(prose)
    if not shouted:
        prose = _LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), prose)
    tokens = _WORD.findall(prose)
    if len(tokens) < 4:
        return None
    prof = ref_latin_profile(prose)
    language = prof["language"]
    if language in (None, "en"):
        return None
    if shouted and not _shouted_evidence(tokens, language, float(prof["diacritic_rate"])):
        return None
    if len({w.lower() for w in tokens} & _STOP.get(language, set())) < 2:
        return None
    return language


rng = random.Random(20260925)
POOLS = [range(0x20, 0x7F), range(0xC0, 0x250), range(0x370, 0x400), range(0x400, 0x500), range(0x900, 0x980),
         range(0x4E00, 0x4F00), range(0x600, 0x700), range(0x300, 0x370), range(0x1E00, 0x1F00),
         range(0xFF21, 0xFF5B), range(0x3040, 0x30FF), range(0x1F600, 0x1F650)]
WORDS = sorted({w for words in _STOP.values() for w in words}) + ["hello", "refund", "INV-2291", "github.com",
                                                                 "user@acme.com", "v1.2.3", "Nav/Com", "OS/2"]


def soup():
    """Word-list prose, sometimes with other scripts, identifiers or capitals mixed in."""
    words = [rng.choice(WORDS) for _ in range(rng.randint(0, 30))]
    if rng.random() < 0.3:
        words = [w.upper() if rng.random() < 0.3 else w.capitalize() if rng.random() < 0.3 else w for w in words]
    text = " ".join(words)
    if rng.random() < 0.4:
        pool = rng.choice(POOLS)
        text += " " + "".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 12)))
    return text


def noise():
    pools = [rng.choice(POOLS) for _ in range(rng.randint(1, 3))] + [POOLS[0]]
    return "".join(chr(rng.choice(rng.choice(pools))) if rng.random() > 0.15 else rng.choice(" \n.,;=()_/\\-@")
                   for _ in range(rng.randint(0, 200)))


texts = [soup() for _ in range(3000)] + [noise() for _ in range(3000)]
texts += ["", "The customer was charged twice", "Set α to 0.05", "Влади́мир called", "मुझसे शुल्क लिया गया",
          "Nos cobraron dos veces en marzo por favor", "İstanbul'da kargo gecikti", "ASCII ONLY SHOUTING HERE NOW"]

mismatch = {"script counts": 0, "letter total": 0, "latin_profile": 0, "prose language": 0}
for text in texts:
    counts = lang._script_counts(text)
    mismatch["script counts"] += counts != ref_script_counts(text)
    mismatch["letter total"] += sum(counts.values()) != sum(ch.isalpha() for ch in text)
    mismatch["latin_profile"] += lang.latin_profile(text) != ref_latin_profile(text)
    for line in text.split("\n"):
        mismatch["prose language"] += lang._named_prose_language(line) != ref_named_prose_language(line)
for name, count in mismatch.items():
    check("fast paths/%s match the per-character code on %d texts" % (name, len(texts)), count, 0)

ascii_text = "Hi, we were billed twice for March. Invoice INV-2291 shows two charges of $49."
check("fast paths/ascii text takes the counting shortcut", lang._script_counts(ascii_text),
      {"latin": sum(ch.isalpha() for ch in ascii_text)})
check("fast paths/no diacritic is ascii", [c for c in _NON_EN_DIACRITICS if c.isascii()], [])
check("fast paths/combining marks are not ascii letters", unicodedata.combining("a"), 0)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all language fast-path tests passed")
sys.exit(1 if FAIL else 0)
