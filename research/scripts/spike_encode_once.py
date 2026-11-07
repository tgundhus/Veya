"""Spike: how much compute would "encode the state once for all questions" save?

Today every question is its own row,

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

so k questions put the whole state through the encoder k times, and `DecisionModel.forward`
then runs its 2 head layers over each full row. This script times that against a *late fusion*
variant built from the model's own modules (encoder, type_emb, head, scorer, act_head):

  1. the encoder runs once over `[CLS] <state> [SEP]` (one row), and once over the k question
     heads `[CLS] ... [SEP] [MASK] opt ... [SEP]` without the state (k short rows, batched);
  2. per question, the head's hidden states and the state's hidden states (its [CLS] dropped, so
     the row has exactly today's length) are concatenated along the sequence, the type embedding
     is added, and the model's own head layers run with a padding mask that covers the head's
     padding. The head comes first, so the [MASK] marker positions are unchanged;
  3. gather at the markers, scorer and act head exactly as `DecisionModel.forward` does.

Question-to-state interaction then happens only in the 2 head layers, so the answers are
meaningless until a model is trained this way. This is about compute only: the timings here are
of random-weight checkpoints with the published shapes, and the answers are not compared.

The variant NOT implemented here: an asymmetric attention mask inside the encoder
------------------------------------------------------------------------------------
Pack one sequence per state, `[CLS] <state> [SEP] | head_1 | head_2 | ... | head_k`, with a
block mask: state tokens attend only to state tokens; the tokens of head_i attend to head_i and
the state (never to another head). State tokens do not depend on any question, so their keys and
values are computed once per layer and shared by every question, while question-to-state
interaction is kept in every encoder layer (as today), not just the 2 head layers.

Its cost, from the same counts the script prints (N state tokens, h_i head tokens):
  * linear work (QKV/out projections, MLP) in the encoder: N+2 + sum(h_i) tokens -- the same as
    late fusion (`encoder_tokens.late`), against k*(N+1) + sum(h_i) today;
  * attention score/value work per global layer: (N+2)^2 + sum(h_i * (h_i + N+2)) query-key
    pairs (`attn_pairs.asym`), which is more than late fusion's (N+2)^2 + sum(h_i^2) but still far
    below today's sum((h_i+N+1)^2); at these lengths attention is a few percent of an encoder
    layer's FLOPs, so the asymmetric variant should cost about what late fusion costs;
  * the head layers can stay as they are (k rows of h_i + N+1 tokens), same as late fusion.
The catch is implementation, not compute: it needs a custom 4-D (or FlexAttention block) mask
instead of the padding mask SDPA gets today, ModernBERT's sliding-window local layers and RoPE
positions have to be made independent of which question follows the state (e.g. state first at
fixed positions, each head at positions after it), and it also needs retraining.

Usage (full sweep; each checkpoint runs its own sweep, english capped at what max_len allows):

    PYTHONPATH=. python research/scripts/spike_encode_once.py \\
        --checkpoint english=/path/to/english --checkpoint multilingual=/path/to/multilingual \\
        --repeats 7 --out spike_encode_once.json

`--quick` runs a tiny sweep (state 20 and 100 tokens, 1 and 3 questions, 2 repeats) as a smoke
test. Timing is of the model forward only; tokenization and collation happen outside the clock.
Rounds are interleaved A B B A after a warm-up, and medians are reported.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_stages import EMAIL, TEN  # noqa: E402
from veya.agent import Agent, _amp_context  # noqa: E402
from veya.common import QTYPES, build_sequence, collate_items, encode_text, render_options  # noqa: E402
from veya.common import _question_head  # noqa: E402

STATE_LENGTHS = (20, 100, 300, 800)
QUESTION_COUNTS = (1, 3, 10)
QUICK_STATE_LENGTHS = (20, 100)
QUICK_QUESTION_COUNTS = (1, 3)


def questions(agent, k):
    """The first k of the ten-question set (choice, score, noul, then more noul/choice)."""
    ids = list(TEN)[:k]
    for qid in ids:
        agent._check_question(qid, TEN[qid])
    return ids, {qid: agent._to_internal(TEN[qid]) for qid in ids}


def state_ids(agent, n):
    """n token ids of a realistic support email, repeated as often as needed."""
    tok = agent.tok
    one = encode_text(tok, EMAIL.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    ids = list(one)
    while len(ids) < n:
        sep = encode_text(tok, "\n\n", add_special_tokens=False)["input_ids"]
        ids += sep + list(one)
    return ids[:n]


def fit_state_len(agent, internal, n):
    """The largest state length <= n that fits every question's row within max_len."""
    tok, max_len, head_max_len = agent.tok, agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    longest = max(len(_question_head(tok, q, head_max_len)[0]) for q in internal.values())
    return min(n, max_len - longest - 1)


def today_batch(agent, ids, internal, st):
    """Today's rows, built exactly as `_encode_state` builds them (state tokenized once)."""
    max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    items = []
    for qid in ids:
        q = internal[qid]
        seq, markers = build_sequence(agent.tok, None, q, max_len, head_max_len, state_ids=st)
        assert len(markers) == len(render_options(q)), qid
        assert len(seq) == len(_question_head(agent.tok, q, head_max_len)[0]) + len(st) + 1, "state truncated"
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return collate_items([items], agent.tok.pad_token_id)


def late_batch(agent, ids, internal, st):
    """The k question heads (no state) as one padded batch, plus the one state row."""
    tok, head_max_len = agent.tok, agent.cfg.get("head_max_len", 192)
    items = []
    for qid in ids:
        q = internal[qid]
        head, markers, _ = _question_head(tok, q, head_max_len)
        items.append({"ids": list(head), "markers": list(markers), "qtype": QTYPES[q["t"]]})
    heads = collate_items([items], tok.pad_token_id)
    srow = torch.tensor([[tok.cls_token_id] + list(st) + [tok.sep_token_id]])
    heads["state_ids"] = srow
    heads["state_mask"] = torch.ones_like(srow)
    return heads


def tail(model, h, pad, marker_pos, marker_mask, qtype):
    """`DecisionModel.forward` after the encoder, line for line (inference: no checkpointing)."""
    h = h + model.type_emb(qtype)[:, None, :]
    if model.head is not None:
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = model.scorer(m).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, -1e4)
    p = torch.softmax(logits.detach(), -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    if p.size(-1) >= 2:
        top2 = p.topk(2, -1).values
    else:
        top1 = p.topk(1, -1).values
        top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    pooled = h[:, 0].float()
    act_logits = model.act_head(torch.cat([pooled, feats], -1))
    return logits, act_logits


def today_forward(model, b):
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])


def today_encoder(model, b):
    return model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state


def today_tail(model, b, h):
    return tail(model, h, ~b["attention_mask"].bool(), b["marker_pos"], b["marker_mask"], b["qtype"])


def late_forward(model, b):
    s = model.encoder(input_ids=b["state_ids"], attention_mask=b["state_mask"]).last_hidden_state
    hq = model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    k = hq.size(0)
    s = s[:, 1:].expand(k, -1, -1)                 # drop the state's [CLS]: <state> [SEP]
    h = torch.cat([hq, s], 1)
    pad = torch.cat([~b["attention_mask"].bool(),
                     torch.zeros(k, s.size(1), dtype=torch.bool, device=s.device)], 1)
    return tail(model, h, pad, b["marker_pos"], b["marker_mask"], b["qtype"])


def local_pairs(L, window):
    """Query-key pairs in one sliding-window layer (ModernBERT: |i - j| <= window // 2)."""
    w = window // 2
    return sum(min(L - 1, i + w) - max(0, i - w) + 1 for i in range(L))


def counts(agent, today, late, n):
    ecfg = agent.model.encoder.config
    d, inter, layers = ecfg.hidden_size, ecfg.intermediate_size, ecfg.num_hidden_layers
    every, window = getattr(ecfg, "global_attn_every_n_layers", 1), getattr(ecfg, "local_attention", 10 ** 9)
    n_global = sum(1 for i in range(layers) if i % every == 0)
    n_local = layers - n_global
    head_layers = len(agent.model.head.layers) if agent.model.head is not None else 0

    k = today["input_ids"].size(0)
    row_lens = today["attention_mask"].sum(1).tolist()
    head_lens = late["attention_mask"].sum(1).tolist()
    s = n + 2
    enc_today_real, enc_today_pad = sum(row_lens), today["input_ids"].numel()
    enc_late_real = sum(head_lens) + s
    enc_late_pad = late["input_ids"].numel() + s
    head_tok_today = today["input_ids"].numel()
    head_tok_late = k * (late["input_ids"].size(1) + s - 1)

    def enc_attn(lengths_pairs_global, lengths_local):
        return n_global * lengths_pairs_global + n_local * lengths_local

    pairs_today = enc_attn(sum(L * L for L in row_lens), sum(local_pairs(L, window) for L in row_lens))
    pairs_late = enc_attn(s * s + sum(h * h for h in head_lens),
                          local_pairs(s, window) + sum(local_pairs(h, window) for h in head_lens))
    # asymmetric mask: head_i queries see head_i + state; the local window is approximated by
    # the same windowed count over a head_i + state row minus the state-only part it reuses.
    pairs_asym = enc_attn(s * s + sum(h * (h + s) for h in head_lens),
                          local_pairs(s, window)
                          + sum(local_pairs(h + s, window) - local_pairs(s, window) for h in head_lens))

    # FLOPs (multiply-add = 2): encoder layer = QKV+out 4d^2 + GLU MLP 3*d*inter per token,
    # attention 2*2*d per pair; head layer = 4d^2 + 8d^2 per token plus attention on full rows.
    enc_lin = 2 * (4 * d * d + 3 * d * inter)
    head_lin = 2 * (4 * d * d + 8 * d * d)
    head_pairs = head_layers * sum(L * L for L in row_lens)

    def gf(enc_tok, pairs, h_tok):
        return round((layers * enc_lin * enc_tok + 4 * d * pairs + head_layers * head_lin * h_tok
                      + 4 * d * head_pairs) / 1e9, 2)

    return {
        "k": k, "state_tokens": n, "head_tokens": head_lens, "row_tokens": row_lens,
        "encoder_tokens": {"today": enc_today_real, "today_padded": enc_today_pad,
                           "late": enc_late_real, "late_padded": enc_late_pad,
                           "asym": enc_late_real},
        "head_layer_tokens": {"today": head_tok_today, "late": head_tok_late, "asym": head_tok_today},
        "encoder_attn_pairs": {"today": pairs_today, "late": pairs_late, "asym": pairs_asym},
        "gflops_est": {"today": gf(enc_today_pad, pairs_today, head_tok_today),
                       "late": gf(enc_late_pad, pairs_late, head_tok_late),
                       "asym": gf(enc_late_real, pairs_asym, head_tok_today),
                       "today_head_share": round(head_layers * head_lin * head_tok_today
                                                 / (gf(enc_today_pad, pairs_today, head_tok_today) * 1e9), 3)},
    }


def clock(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def measure(agent, today, late, repeats, warmup):
    model = agent.model
    with torch.no_grad(), _amp_context(agent.device, agent.dtype, agent.amp_enabled):
        h_today = today_encoder(model, today)
        fns = {
            "today": lambda: today_forward(model, today),
            "late": lambda: late_forward(model, late),
            "today_encoder": lambda: today_encoder(model, today),
            "today_tail": lambda: today_tail(model, today, h_today),
        }
        # the split must add up to the real forward, or the shares below describe something else
        ref, split = today_forward(model, today), today_tail(model, today, h_today)
        assert torch.allclose(ref[0], split[0], atol=1e-4) and torch.allclose(ref[1], split[1], atol=1e-4)
        lf = late_forward(model, late)
        assert lf[0].shape == ref[0].shape and lf[1].shape == ref[1].shape
        assert torch.isfinite(lf[0]).all() and torch.isfinite(lf[1]).all()
        for _ in range(warmup):
            for fn in fns.values():
                fn()
        times = {name: [] for name in fns}
        order = list(fns)
        for r in range(repeats):
            for name in (order if r % 2 == 0 else order[::-1]):   # A B B A
                times[name].append(clock(fns[name]))
    med = {name: statistics.median(v) for name, v in times.items()}
    enc, tl = med["today_encoder"], med["today_tail"]
    return {
        "median_ms": {name: round(v * 1e3, 2) for name, v in med.items()},
        "speedup": round(med["today"] / med["late"], 2),
        "today_encoder_share": round(enc / (enc + tl), 3),
        "today_head_share": round(tl / (enc + tl), 3),
        "raw_ms": {name: [round(x * 1e3, 2) for x in v] for name, v in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                        help="checkpoint to sweep; repeat for several")
    parser.add_argument("--repeats", type=int, default=None, help="timed rounds per point (default 7, quick 2)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--quick", action="store_true", help="tiny sweep for a smoke test")
    parser.add_argument("--states", help="comma-separated state lengths, overriding the sweep's")
    parser.add_argument("--questions", help="comma-separated question counts (1-10), overriding the sweep's")
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    repeats = args.repeats or (2 if args.quick else 7)
    lengths = QUICK_STATE_LENGTHS if args.quick else STATE_LENGTHS
    ks = QUICK_QUESTION_COUNTS if args.quick else QUESTION_COUNTS
    if args.states:
        lengths = tuple(int(x) for x in args.states.split(","))
    if args.questions:
        ks = tuple(int(x) for x in args.questions.split(","))

    results = {"torch": torch.__version__, "threads": torch.get_num_threads(), "repeats": repeats,
               "warmup": args.warmup, "checkpoints": {}}
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            parser.error("--checkpoint takes NAME=PATH, got %r" % spec)
        agent = Agent(path, device="cpu")
        agent.model.eval()
        points = []
        print("\n== %s  (max_len %s, head_max_len %s, device %s, amp %s)" % (
            name, agent.cfg.get("max_len"), agent.cfg.get("head_max_len"), agent.device, agent.amp_enabled))
        print("%5s %3s | %11s %11s | %11s %11s | %9s %9s %7s | %7s %6s" % (
            "state", "k", "enc today", "enc late", "head today", "head late",
            "today ms", "late ms", "speedup", "enc%", "GF x"), flush=True)
        for n_req in lengths:
            for k in ks:
                ids, internal = questions(agent, k)
                n = fit_state_len(agent, internal, n_req)
                st = state_ids(agent, n)
                today, late = today_batch(agent, ids, internal, st), late_batch(agent, ids, internal, st)
                point = {"state_tokens_requested": n_req, **counts(agent, today, late, n)}
                try:
                    point.update(measure(agent, today, late, repeats, args.warmup))
                except Exception as e:  # keep the sweep going; the error is part of the result
                    point["error"] = "%s: %s" % (type(e).__name__, e)
                    print("%5d %3d | ERROR %s" % (n, k, point["error"]))
                    points.append(point)
                    continue
                points.append(point)
                et, ht, g = point["encoder_tokens"], point["head_layer_tokens"], point["gflops_est"]
                print("%5d %3d | %11d %11d | %11d %11d | %9.1f %9.1f %6.2fx | %6.1f%% %5.2fx" % (
                    n, k, et["today_padded"], et["late_padded"], ht["today"], ht["late"],
                    point["median_ms"]["today"], point["median_ms"]["late"], point["speedup"],
                    100 * point["today_encoder_share"], g["today"] / g["late"]), flush=True)
        results["checkpoints"][name] = {"path": path, "max_len": agent.cfg.get("max_len"),
                                        "head_max_len": agent.cfg.get("head_max_len"), "points": points}
        del agent
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
