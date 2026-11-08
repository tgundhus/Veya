"""The cached question head must build exactly the rows the uncached code built.

`build_sequence` keeps the tokenized `[CLS] <type> instructions [SEP] [MASK] opt ... [SEP]` part
per tokenizer and question, and only appends the state per request. `reference_build_sequence`
below is the code before that change, kept verbatim. Seeded random questions of every type, with
long options, tight head budgets, option orders, both truncation directions and pre-tokenized
states, must give identical ids and markers on a cold and on a warm cache, through a real fast
tokenizer and through a plain fake one.

Run: python tests/test_question_heads.py
"""
import os
import random
import sys
import weakref

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from veya import common  # noqa: E402
from veya.common import build_sequence, encode_text, render_options, serialize_state  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def reference_build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None,
                             truncate_left=False, state_ids=None):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = encode_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_tokens = encode_text(tok, " " + opts[i].replace(mask_tok, " "), add_special_tokens=False,
                                 truncation=True, max_length=48)["input_ids"]
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_text(tok, serialize_state(state).replace(mask_tok, " "),
                                add_special_tokens=False)["input_ids"]
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


WORDS = ["w%d" % i for i in range(300)]
vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
for w in WORDS + ["question", ":", "choice", "score", "noul", "level", "yes", "no", "true", "false"]:
    vocab.setdefault(w, len(vocab))
backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
backend.pre_tokenizer = Whitespace()
FAST = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                               cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


class _Tok:
    __slots__ = ()
    mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = "[MASK]", 4, 2, 3, 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (hash(w) % 50) for w in text.split()]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


class FakeTok(_Tok):
    """No __slots__ of its own, so it can be weakly referenced and its heads are kept."""


class SlottedTok(_Tok):
    """No __weakref__ slot anywhere in its bases: the head cannot be kept for it, and must still be right."""
    __slots__ = ()


rng = random.Random(7)


def text(lo, hi):
    words = [rng.choice(WORDS) for _ in range(rng.randint(lo, hi))]
    if words and rng.random() < 0.1:
        words.insert(rng.randrange(len(words)), "[MASK]")
    return " ".join(words)


def question():
    t = rng.choice(["choice", "score", "noul"])
    q = {"t": t, "ins": text(1, 30)}
    if t == "choice":
        q["crit"] = {"opt%d" % i: (text(0, 70) if rng.random() < 0.7 else None) for i in range(rng.randint(1, 12))}
    elif t == "score":
        q["crit"] = [text(1, 20) for _ in range(rng.randint(1, 6))]
    else:
        q["crit"] = rng.choice([None, {"true": text(1, 10)}, {"false": text(1, 10), "true": text(1, 10)}])
        if rng.random() < 0.3:
            q["labels"] = {"false": "B", "true": "A"}
    return q


mismatch = 0
cases = 0
for tok in (FAST, FakeTok(), SlottedTok()):
    for _ in range(600):
        q = question()
        n = len(render_options(q))
        order = rng.sample(range(n), n) if rng.random() < 0.3 else None
        kw = {"max_len": rng.choice([24, 64, 128, 512]), "head_max_len": rng.choice([12, 16, 32, 64, 192]),
              "option_order": order, "truncate_left": rng.random() < 0.5}
        state = text(0, 200) if rng.random() < 0.8 else {"subject": text(1, 5), "body": text(0, 80)}
        want = reference_build_sequence(tok, state, q, **kw)
        for _warm in range(2):
            cases += 1
            mismatch += build_sequence(tok, state, q, **kw) != want
        ids = encode_text(tok, serialize_state(state).replace(tok.mask_token, " "),
                          add_special_tokens=False)["input_ids"]
        cases += 1
        mismatch += build_sequence(tok, state, q, state_ids=ids, **kw) != want
check("heads/cold and warm rows equal the uncached code (%d builds)" % cases, mismatch, 0)

check("heads/kept per tokenizer", FAST in common._QUESTION_HEADS, True)
check("heads/bounded per tokenizer", len(common._QUESTION_HEADS[FAST]) <= common._QUESTION_HEADS_MAX, True)


def weakly_referenceable(obj):
    try:
        weakref.ref(obj)
    except TypeError:
        return False
    return True


# The premise of the SlottedTok rows above: its heads cannot be kept, so every one of them went
# through the uncached fallback, while FakeTok's went through the cache.
check("heads/SlottedTok cannot be weakly referenced", weakly_referenceable(SlottedTok()), False)
check("heads/FakeTok can", weakly_referenceable(FakeTok()), True)


class Counting(FakeTok):
    def __init__(self):
        self.calls = 0

    def __call__(self, text, **kw):
        self.calls += 1
        return super().__call__(text, **kw)


counting = Counting()
q = {"t": "choice", "ins": "pick one", "crit": {"a": "x", "b": "y", "c": "z"}}
build_sequence(counting, "first state", q)
first = counting.calls
build_sequence(counting, "second state", q)
check("heads/first build tokenizes instructions, 3 options and the state", first, 5)
check("heads/a repeat question only tokenizes the new state", counting.calls - first, 1)

other = Counting()
build_sequence(other, "first state", q)
check("heads/another tokenizer does not reuse them", other.calls, 5)

rows = [tuple(build_sequence(counting, "s", q)[0]) for _ in range(2)]
first_row = build_sequence(counting, "s", q)[0]
first_row.append(-1)
check("heads/returned rows are fresh lists", tuple(build_sequence(counting, "s", q)[0]), rows[0])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all question head tests passed")
sys.exit(1 if FAIL else 0)
