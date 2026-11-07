"""Time `Agent._encode_state` with and without the per-tokenizer question-head cache.

"Before" is `build_sequence` as it was before the cache, kept verbatim below; "after" is the
current one. Both run through the real `_encode_state` with the checkpoint's tokenizer, in
alternating A B B A rounds, and must produce identical rows. No weights are loaded.

    python research/scripts/bench_question_heads.py --checkpoint /path/english --out heads.json
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import veya.agent as agent_mod  # noqa: E402
from bench_stages import EMAIL, TEN, TICKET, TRIAGE  # noqa: E402
from veya.agent import Agent, _load_tokenizer  # noqa: E402
from veya.common import encode_text, render_options, serialize_state  # noqa: E402


def uncached_build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None,
                            truncate_left=False, state_ids=None):
    """`build_sequence` before the question-head cache, verbatim."""
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


CASES = {
    "ticket, 3 questions": ([TICKET], TRIAGE),
    "ticket, 10 questions": ([TICKET], TEN),
    "email, 3 questions": ([EMAIL], TRIAGE),
    "batch of 32 tickets, 3 questions": ([TICKET + " #%d" % i for i in range(32)], TRIAGE),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory with tokenizer/ and its config")
    parser.add_argument("--rounds", type=int, default=14)
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args()

    with open(os.path.join(args.checkpoint, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    agent = Agent.__new__(Agent)
    agent.cfg = cfg
    agent.tok = _load_tokenizer(os.path.join(args.checkpoint, "tokenizer"), cfg)
    cached = agent_mod.build_sequence

    results = {}
    for label, (states, questions) in CASES.items():
        ids = list(questions)
        internal = {q: Agent._to_internal(questions[q]) for q in ids}

        def encode_all():
            return [agent._encode_state(s, ids, internal) for s in states]

        agent_mod.build_sequence = uncached_build_sequence
        before_rows = encode_all()
        agent_mod.build_sequence = cached
        if encode_all() != before_rows:
            raise SystemExit("%s: rows differ between the cached and uncached paths" % label)
        n = max(20, 2000 // (len(states) * len(ids)))
        times = {"before": [], "after": []}
        for i in range(args.rounds):
            for which in (("before", "after") if i % 2 == 0 else ("after", "before")):
                agent_mod.build_sequence = uncached_build_sequence if which == "before" else cached
                start = time.perf_counter()
                for _ in range(n):
                    encode_all()
                times[which].append((time.perf_counter() - start) / n)
        agent_mod.build_sequence = cached
        before = statistics.median(times["before"]) * 1e3
        after = statistics.median(times["after"]) * 1e3
        results[label] = {"rows": len(states) * len(ids), "encode_ms_before": round(before, 3),
                          "encode_ms_after": round(after, 3), "speedup": round(before / after, 2)}
        print("%-34s rows=%3d  before %8.3f ms  after %8.3f ms  x%.2f" % (
            label, len(states) * len(ids), before, after, before / after), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
