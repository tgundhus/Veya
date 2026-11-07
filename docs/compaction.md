---
title: Experimental context compaction
description: Opt-in local pruning of replayable tool history, with conservative retention and rollback.
---

# Experimental context compaction

`ContextCompactor` uses a local Veya predictor to prune older tool calls and
results. It does not summarize conversation text. Nothing invokes it automatically;
your application explicitly supplies a transcript and decides whether to use the
returned copy. The original transcript remains unchanged and should be retained.

This is an experimental retention policy. Low relevance probabilities are not
proof that information can be discarded. Validate the policy on your own tasks
before using it in an agent loop.

## Local usage

```python
from veya import Agent, ContextCompactor

# A directory containing an already downloaded checkpoint; inference stays local.
agent = Agent("/models/veya", device="cpu", performance_mode="default")
compactor = ContextCompactor(agent)

messages = [
    {"role": "system", "text": "Never edit generated files."},
    {"role": "assistant", "text": "", "toolUses": [{
        "tool_use_id": "read-old", "tool": "Read",
        "input": {"file_path": "docs/old.md", "snapshot": "immutable-revision"},
        "replayable": True,
    }]},
    {"role": "user", "text": "", "toolResults": [{
        "tool_use_id": "read-old", "text": "Old documentation contents...",
    }]},
    # ...later messages...
]
result = compactor.compact(
    messages,
    goal="Fix src/login.py. The old documentation is unrelated.",
    constraints="Keep generated files unchanged.",
)
print(result["decisions"], result["stats"])
compacted_messages = result["messages"]
```

The first message and newest six messages are protected. The short example above
therefore stays intact. A longer transcript can have eligible older pairs.
The compactor never runs tools or saves/deletes transcript files.

An existing `ONNXAgent` works through the same batch API. With a `Router`, pass
`model="english"` (or your registered checkpoint name) to the compactor to pin
selection. Register local checkpoint directories for offline routing. The module
does not create a predictor, request a hosted service or add a runtime dependency;
the caller is responsible for supplying a local predictor and preloading artifacts
for offline operation.

## Transcript and protection contract

Messages use `role`, `text`, and optional `toolUses` / `toolResults` lists.
Calls have `tool_use_id`, `tool`, and an `input` dictionary. Results have the
matching `tool_use_id` and `text`. IDs must be unique within calls and within
results; a result must follow its call. Unfinished calls are kept. Extra JSON
metadata survives edits, and the returned transcript does not share mutable
objects with the input. Native provider formats require an adapter; opaque
provider handles must be rebuilt by that adapter when content changes.

Only a call explicitly marked `replayable=True` can be pruned. This flag asserts
that the exact result can be recovered from the same immutable snapshot without
side effects. A generic file read against a mutable working directory does not
establish replayability. Do not mark writes, deployment receipts, expiring query
results, or unique observations as replayable.

The following are always retained:

- First and recent messages, system/developer messages, and all conversation text.
- Messages, calls, or results marked `pinned=True`.
- Calls or results marked `isError=True`.
- Incomplete calls and blocks/messages with a `content` field, including multimodal content.
- Results for which the predictor does not confirm an untruncated scoring state.

Only tool blocks are removed. A message emptied by removing a pair can disappear;
messages carrying text or extra metadata remain. A result is never left without
its call. Native provider message validity remains the adapter's responsibility.

## Scoring and fallback

Each eligible pair gets two `noul` judgments: whether the call still matters and
whether the full result is still needed. The scoring state includes the explicit
goal and constraints, first/system/developer/pinned message text, recent message
text, the candidate's input, and its output. Outputs longer than 600 characters
are represented by the first and last 300 characters. This is bounded evidence
selection, not a full-history relevance judgment: cross-turn dependencies and
information in omitted output middles can be missed.

The default `drop_threshold=0.1` requires a result retention probability at or
below 0.1 before shortening anything. If both retention probabilities are at or
below that threshold, the pair is removed. Otherwise only the result is shortened
to its first 300 characters plus an omission marker. Short outputs are not enlarged.
Higher or uncertain result probabilities keep the pair intact. Thresholds need
task-specific evaluation; these checkpoints were not validated as compaction models.
Either judgment marked `low_confidence`, `abstained` or `unevaluated` also keeps
the whole pair. An explicit unrecognized abstention state cannot authorize
pruning. These pairs report `unverified_confidence` in their decision reason;
predictors without confidence-gate annotations retain the probability-only policy.

Candidates are scored in batches of eight states (two question rows per state).
`max_len=None` retains the predictor's checkpoint budget. Increasing `max_len`
does not establish long-context accuracy. Any model-reported truncation retains
the affected pair; missing truncation metadata also retains it. The compactor does
not use sliding-window maximum aggregation for these relevance judgments.

A failed batch or invalid probability rolls back all edits. The output also rolls
back unless reduction reaches `min_reduction_ratio=0.1` and reduces size. The
`stats.fallback` field explains rollback; scoring error messages are excluded from
diagnostics because they can contain private input. Caller input/schema errors
raise directly. Model scores remain in `decisions` for inspection even after rollback.

Statistics report canonical JSON character counts, elapsed time, batch/candidate
counts and reduction. Supply `token_counter(messages)` to get downstream token
counts and make the reduction gate use tokens instead of characters. The callback
must count the whole supplied transcript, including tool inputs and outputs, using
the target model's formatting. Character savings alone do not establish token savings.

## Validation and benchmark

The [5 November checkpoint report](reports/compaction-2026-11-05.md) records the
initial local run. Conservative defaults preserved all annotated evidence but
removed no tokens; this run does not establish useful model-driven compaction.

Run the deterministic regression suite without checkpoints:

```bash
python tests/test_compaction.py
```

The suite is included in the shared CI/release test list.

Run the offline benchmark using existing local checkpoints:

```bash
python research/scripts/bench_compaction.py \
  --checkpoint /models/veya \
  --checkpoint /models/laya-multilingual \
  --out research/results/compaction.json
```

The harness compares retaining everything, the compactor's conservative defaults,
and a naive policy that deletes old completed pairs outside the protected window.
It records checkpoint hashes, individual probabilities/actions, exact annotated
evidence retention, token counts and warm CPU timings over three repetitions.
It sets offline flags and accepts only local weight directories. It uses the
checkpoint tokenizer over canonical transcript JSON, not a downstream chat template.
CPU autocast is explicitly disabled for this fp32 benchmark, including when the
parent process sets `VEYA_CPU_AMP=bf16`; the runtime precision is checked after loading.

The twelve synthetic cases cover stale reads/searches, relevant source, exact
errors, non-repeatable receipts, pins, recency, evidence in output middles, required
call history, oversized goals and instructions embedded in tool output. These are
small regression examples, not an accuracy dataset or a measurement of downstream
agent task success. No hosted model is involved.
The injection payload is placed in the visible output head; its presence in the
scoring state is covered by the regression suite. Evidence in output middles is
tested separately.
