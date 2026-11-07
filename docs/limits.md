# Limits

What the models and Veya do not do well, measured where it could be. [Benchmarks and known
limits](benchmarks.md) explains how the benchmarks were run, and
[BENCHMARKS.md](https://github.com/tgundhus/Veya/blob/main/BENCHMARKS.md) has every run.

## Veya's cache and servers

- **The cache replays exact repeats only.** Requests that mean the same but differ in form are new
  requests: reversing the option order changed 19-36% of decisions, and one trailing space changed
  up to 6.6%. Normalise requests before they reach the model if you need those to agree.
- **`veya-serve` and the MCP server do not install the cache yet.** Build the app yourself to get
  one: `create_app(router=Router(hooks=[DecisionCache(...)]))` from `veya.serve`.
- **In that server, a hit waits for other clients' misses.** It runs every request behind one lock,
  so a 0.7 ms hit took 169 ms while three other clients sent misses.
- **Two threads predicting at once on an Apple GPU (MPS) abort the process.** PyTorch's MPS backend
  is not safe for concurrent forward passes; run inference on one thread there. The cache's
  coalescing makes concurrent *identical* requests safe, since only one of them runs the model.
- **int8 changes decisions.** On Apple silicon it changed 10-36% of decisions and was no faster than
  fp32 (ONNX) or 2.8x slower (PyTorch); PyTorch int8 lost 10.5 points of accuracy on MASSIVE
  English. A `DecisionCache` shared by fp32 and int8 replays the first decision to both.
- **A model upgrade starts a new memory.** The cache's key includes the model's fingerprint
  (version, checkpoint, revision and temperatures), so decisions stored before an upgrade are not
  replayed after it, unless you pin `fingerprint=`.

Measurements: [Real checkpoints on an Apple M4 Max](reports/real-checkpoints-m4-max.md).

## Where Jev leads

* **High-cardinality label spaces (>20 options at default settings):** On Banking77, Jev scores 0.870 (on 72 labels) while Veya scores 0.425 (on 77 labels at default 256-token head budget). This is an architectural token-budget constraint: options share a fixed `head_max_len` budget (192 tokens on English, 256 on multilingual), so 77 options receive only ~3 to 4 tokens per label, causing text to become indistinguishable. Jev supports up to 255 options out-of-the-box. While `laya-multilingual` supports 1,024 context (and up to 8,192 in the encoder) and you can raise `agent.cfg["head_max_len"] = 512` at runtime, Jev is currently better suited for 50+ options in a single prompt without tuning. `predict_shortlist` (see [Honest limits](#known-limits)) keeps the top `k` labels with a caller-supplied embedding, then runs one forward pass on that shortlist.
* **Soft distribution matching:** On typed-decisions, while Veya achieves higher argmax accuracy (0.766 vs 0.727), Jev achieves higher soft accuracy (0.580 vs 0.471) against the teacher's full probability distributions.
* **Out-of-the-box raw calibration:** Before temperature scaling, the base checkpoint has higher raw ECE (0.213 vs 0.144). Veya achieves its 0.081 ECE after domain temperature fitting.

Full detail, including every workflow and all 51 languages: **[`BENCHMARKS.md`](https://github.com/tgundhus/Veya/blob/main/BENCHMARKS.md)**.

## Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`veya`) and
**0.314 -> 0.106** (`laya-multilingual`). `laya-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

At checkpoint load, numeric temperature entries are clamped to `[0.5, 5.0]`; invalid or
non-finite entries use the neutral fallback `1.0`. A runtime warning reports the affected
entries and applied values. Bucket-specific temperatures still take precedence over per-type
values, including when a bucket uses the fallback. Raw values remain available in
`agent.temperature_raw` and `agent.temperature_by_options_raw`. A fallback prevents a loading
failure; it does not establish calibrated confidence.

## Known limits

* **The base checkpoints are near chance on typed-decisions zero-shot** -- 0.362 and 0.352
  against a 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split. Veya is a fast base to
  specialise, not a zero-shot decision engine.
* **Avoid boolean-word labels in `choice` questions.** Choice keys are rendered verbatim, and the
  current checkpoints can follow labels such as `true`/`false` or `yes`/`no` instead of the option
  descriptions. Use semantic labels or opaque labels such as `A`/`B`, and validate them on the
  checkpoint and states you serve.
* **Semantic `choice` labels do not make negation safe.** In the five cancellation examples from
  Veya issue #377, a CPU run on Veya 0.3.20 with
  `no_action` / `cancel_account` keys selected `cancel_account` for all four negated requests on
  `veya` and two on `laya-multilingual`; one multilingual answer assigned it probability `0.9998`.
  The positive control passed on both checkpoints. These are narrow cancellation examples, not
  evidence that every negated state fails. Validate the exact checkpoint and wording you serve;
  using semantic keys alone does not avoid this failure.
* **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget. **`head_max_len` is a cap, not the head length.** The renderer fills the option prompt budget only as far as the question and its option descriptions need, and `build_sequence` then sizes the state from the head it *actually built* (`room = max_len - head_len - 1`). The real room therefore depends on the question, not on the cap:
  * `veya` (English) defaults to 512 context (`head_max_len = 192`). The shipped department question above renders a **48-token** head, leaving **463 tokens for state**, not 320.
  * `laya-multilingual` and `laya-typed-decisions` default to 1,024 context (`head_max_len = 256`). The shipped question renders a **45-token** head, leaving **978 tokens for state**, not 768 (mmBERT-base encoder supports up to 8,192 with RoPE).
  `max_len - head_max_len` is **not a safe figure in either direction**, so size a state to `max_len - head_len - 1` for the question you are actually asking:
  * For a question that does not fill the cap it **understates** the room, as above — conservative, but it discards state the model would have read.
  * For a high-cardinality question it **overstates** it. `per = max(4, (head_max_len - 16) // k)` floors at 4 tokens per option and `head_ids` keeps a floor of 8, so past roughly `head_max_len / 4` options the head grows *beyond* the cap. A state sized to the cap-based figure is then **truncated**, not merely conservative:

    | | `k` options | head | real room | cap-based figure |
    |---|---|---|---|---|
    | `veya`, `max_len=512` | 77 | 319 | **192** | 320 |
    | `veya`, `max_len=512` | 100 | 411 | **100** | 320 |
    | `laya-multilingual`, `max_len=1024` | 100 | 411 | **612** | 768 |

    The crossover is at `k ≈ head_max_len / 4` and the gap widens from there; `MAX_CHOICE_OPTIONS = 100` in `veya/serve.py` puts the overstating regime within reach over HTTP.

  At default settings, a 77-option question like Banking77 allocates only `(256 - 16) // 77` ≈ 3–4 tokens per label, which causes accuracy to fall off sharply (0.425 vs Jev's 0.870). If evaluating 50+ options in a single question:
  1. Raise `agent.cfg["head_max_len"] = 512` and `agent.cfg["max_len"] = 1024` (or up to 2048 / 4096 / 8192) so every option has enough tokens to remain distinct. Both are also per-request: `predict(state, questions, head_max_len=512, max_len=1024)` widens one question without changing the agent for everyone else, and every LangChain node takes the same two arguments ([LangChain guide](langchain.md)). `veya --questions` takes the same two budgets as `--max-len` / `--head-max-len`.
  2. Or shortlist with embeddings and run one forward pass on the top `k` labels (`predict_shortlist`, example below). `predict` and `system_one` still score every criterion they are given.
  3. Or split the label set yourself into a coarse question and a fine question.
  4. Or let the decision model narrow the set itself: `veya.predict_tournament(agent, state, questions)` answers the labels in groups of at most 16, every group in the same forward pass, then asks the question again over the group winners, so up to 256 labels take two `predict` calls and no embedder. On the full test splits with the English checkpoint it took BANKING77 (77 labels) from 0.430 to 0.610, CLINC150 (150) from 0.625 to 0.876 and MASSIVE intent (60) from 0.515 to 0.569, with ECE between 0.06 and 0.14 instead of 0.30 to 0.37, at about twice the latency of one question ([`research/benchmarks/tournament`](https://github.com/tgundhus/Veya/blob/main/research/benchmarks/tournament/README.md)). When the labels fit uncut in a raised budget, as MASSIVE's 60 do at `head_max_len=512`, raising it did better (0.622 against 0.590 on 500 rows). Probabilities on a tournament choice are over its finalists.

```python
import veya

questions = {
    "intent": {
        "type": "choice",
        "instructions": "Which banking intent is this?",
        "criteria": {
            "card_arrival": "where is my card",
            "transfer_fee": "fee charged on a transfer",
            # ...the rest of a large label set
        },
    }
}
result = veya.predict_shortlist(
    agent,
    {"text": "I was charged twice for a transfer"},
    questions,
    embed_fn=veya.embed_fn_from_agent(agent),  # or any callable: texts -> (n, dim)
    k=20,
)
result["shortlist"]["intent"]["labels"]  # the top 20 labels sent to the model
```

`embed_fn(texts)` returns one vector per string. `embed_fn_from_agent` mean-pools the encoder already loaded on the agent; the decision head runs in the following `predict` / `system_one` call. Probabilities on a shortlisted choice are over those `k` labels. When `k` is at least the number of labels, the original question is passed through and `embed_fn` is not called. `veya.shortlist_choice` returns bare labels; pass `return_scores=True` for the `(labels, scores)` pair in rank order (`None` when nothing was dropped), the same cosines `predict_shortlist` reports in its `shortlist` metadata.

Shortlisting the same option set on every request re-embeds option texts that do not change. Wrap the embedder once with `veya.cached_embed_fn(embed_fn)` and repeat calls embed only the new query text: lookups are exact string matches into an LRU of at most 4,096 entries (about `maxsize * dim * 4` bytes, so ~12 MB at the default with a 768-dim encoder), and texts missing from the cache are still embedded in one batched call. The wrapper's `cache_info()` reports hits and misses; call `cache_clear()` if the model behind `embed_fn` changes.

Veya issue #102 reports that a top-20 zero-shot shortlist moved a BANKING77 run from 54.3% to 60.8% on the reporter's setup. Those figures are the reporter's; this repository has not remeasured them.

* Ordinal `score` questions are the weakest primitive (SST-5 0.372).
* **`noul` can follow its option labels instead of the state, most strongly on `veya` (English).** `noul` renders its two options as `false:` / `true:` by default, and on the English checkpoint that label pair can dominate the answer, returning a confident "no" for clearly positive input (#156). Until a retrained checkpoint lands, check `noul` answers on your own data. You can override the model-facing pair while keeping the `noul` result as P(true):

  ```python
  {"type": "noul", "instructions": "Is this review positive?",
   "criteria": {"true": "the review is positive", "false": "the review is negative"},
   "labels": {"true": "A", "false": "B"}}
  ```

  A `noul` with no `criteria` renders one generic option pair for every state (`no, the statement
  does not hold` / `yes, the statement holds`). On `veya` that pair carries the whole decision, so
  it answers "no" whatever the state — give a `noul` criteria if you need it to discriminate. On
  `laya-multilingual` the criteria-less form does read the state.

  Label sensitivity varies by checkpoint and state, so validate the override on your own data. A
  two-option `choice` with neutral keys remains another workaround:

  ```python
  {"type": "choice", "instructions": "Is this review positive?",
   "criteria": {"A": "yes, the review is positive", "B": "no, the review is negative"}}
  ```

  `criteria` on a `noul` must be keyed `true`/`false` — those two keys *are* the option text the
  model reads, so any other key is rejected instead of being quietly replaced with the defaults.
  Before that check, `criteria: {"yes": ..., "no": ...}` was accepted, dropped, and answered
  against `false:` / `true:` anyway, which cost 2 of 3 clearly positive reviews on the English
  checkpoint (Veya issue #156). Use `labels` as above to
  change the wording without touching the option text.
* **`laya-multilingual` has a position bias on `score` questions** (#131): it rarely picks the first-listed level, in any language. For English score questions, route to `model="english"`, and for other languages validate score outputs on your own data before relying on them.
* **`action.act_probability` carries no usable signal yet** (#185). It reads 1.0 for almost every input, and its raw logits run against correctness (AUROC 0.30 on 396 labelled decisions). Gate on `confidence` instead, which reaches an AUROC of 0.77 on the same items.
* `veya` collapses outside English; `laya-multilingual` is weaker on English. Route, or pick
  deliberately.

