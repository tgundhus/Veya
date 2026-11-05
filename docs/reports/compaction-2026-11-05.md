---
title: Experimental compaction, 5 November 2026
description: Conservative local context compaction retained all evidence but achieved no token reduction on a small synthetic suite.
---

# Experimental compaction, 5 November 2026

The corrected `ContextCompactor` run achieved **zero token savings** with its
conservative defaults. It retained all annotated evidence, including evidence
that a naive recency policy deleted. These results justify keeping the feature
experimental and opt-in; they do not demonstrate useful model-driven compaction.

## Method

The [benchmark harness](https://github.com/tgundhus/laya-pro/blob/main/research/scripts/bench_compaction.py)
uses twelve synthetic annotated coding transcripts and existing local weights.
The [raw results](https://github.com/tgundhus/laya-pro/blob/main/research/results/compaction_windows_20261105.json)
record every retention probability, action, fallback, repetition time and artifact hash.
The [original run before review fixes](https://github.com/tgundhus/laya-pro/blob/main/research/results/compaction_windows_20261105_before_review_fixes.json)
is retained separately for comparison.

Two local Hugging Face snapshot directories were available:
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and
`7b928d828b7b0e022f929d9bd2e44165aa270148`. Their English model weights and runtime
configuration hashes are identical. These are repeat measurements of one model,
not independent model comparisons. No multilingual or typed-decisions checkpoint
was available locally; the harness accepts additional local checkpoint directories.

The model SHA-256 is
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
The run used CPU eager inference, fp32, four intra-op threads and one inter-op
thread, one warmup followed by three repetitions per case, without decision
caching. Offline flags prevented model downloads. Workstation load was uncontrolled.
The corrected run explicitly disabled CPU autocast and checked runtime precision.
It was launched with `LAYA_CPU_AMP=bf16` in the parent environment to verify that
the harness still enforces the reported fp32 execution.
The loader warned about an invalid `choice:11+` temperature and clamped that bucket;
this workload only uses two-option `noul` questions, whose calibration was unchanged.

Token counts include canonical JSON transcript text, tool inputs and outputs,
using the checkpoint tokenizer. These are not counts for a downstream provider's
chat template. Exact evidence retention checks search the output for annotated
required strings or required call IDs. They do not run an agent or measure task success.

## Results

The aggregate token counts were identical across the two snapshot-directory runs:

| Policy | Tokens before | Tokens after | Reduction | Cases losing required evidence |
|---|---:|---:|---:|---:|
| Retain original history | 11,392 | 11,392 | 0% | 0 / 12 |
| Conservative compactor | 11,392 | 11,392 | 0% | 0 / 12 |
| Naive old-pair deletion | 11,392 | 1,869 | 83.6% | 8 / 12 |

The naive baseline deletes completed pairs wholly outside the first/newest-six
message window. It does not honor explicit replayability, error protection or
extra pins. It illustrates the cost of indiscriminate pruning, not a competitive
optimized compactor.

Four cases were protected without model inference: an error, a non-repeatable
receipt, a pinned result and a recent result. Of the remaining eight cases, two
had truncated scoring states (a symbol-heavy listing and an oversized goal), so
their model answers could not authorize edits. The six other result-retention
probabilities ranged from 0.3235 to 0.7734, all above the default 0.1 drop threshold.
No case met the pruning policy. Decisions agreed across all three repetitions.

Warm model-scored cases took approximately 1–2 seconds each on this machine;
protected cases took approximately 1–2 milliseconds including token counting.
Consult the raw per-case timings for measurements; these are not production
throughput guarantees or GPU results.

## Review fixes and before/after evidence

The compactor now respects confidence-gate annotations on either retention
judgment. `low_confidence`, `abstained`, `unevaluated`, or an unrecognized explicit
abstention state retains the whole pair with reason `unverified_confidence`.
An actual core confidence-gate regression reproduced three pair deletions before
the fix and zero afterward. Answers explicitly passing the gate can still prune.
The twenty compaction regression tests are registered in the shared CI/release
suite list; the public API contract also checks abstention behavior.

The original prompt-injection example placed the attack in an omitted output
middle, so it did not test a visible attack. The corrected fixture puts it in the
output head, and a regression test checks the actual scoring state's excerpt.
This relocation changes the fixture's aggregate token count by two tokens; the
before/after runs do not have byte-identical inputs.

| Measurement, English snapshot `55cf4c4` | Before fixes | After fixes |
|---|---:|---:|
| Aggregate transcript tokens | 11,394 | 11,392 |
| Compactor token reduction | 0% | 0% |
| Cases losing annotated required evidence | 0 / 12 | 0 / 12 |
| Injection payload visible in scoring excerpt | No | Yes |
| Injection case call-retention probability | 0.5923 | 0.3940 |
| Injection case result-retention probability | 0.5291 | 0.3235 |
| Median case latency | 1.074 s | 1.125 s |

Both snapshot-directory runs produced the same probabilities and actions. The
other eleven cases' retention probabilities were unchanged. Timing differences
on this active workstation do not establish a performance regression or gain.
The visible injection remains retained at the default threshold. One adversarial
example does not establish injection robustness, and its lower retention scores
reinforce the need to evaluate changes to the pruning threshold.

## Interpretation and next checks

The mechanics can prune under controlled scores, and the regression tests cover
pair integrity, input immutability, explicit protections, batch rollback, malformed
answers, context truncation and token-based acceptance. The real checkpoint run
shows conservative retention, not successful discrimination of stale content.

Before deploying automatic pruning, evaluate real transcripts and stronger
retention labels, including dependencies spanning older turns, changed goals,
unique outputs, output-middle evidence, prompt injection and actual downstream
task completion. Compare a carefully engineered deterministic policy as well as
different checkpoints or compaction-specific fine-tuning. Do not raise thresholds
merely to obtain a favorable token-reduction number.

See the [compaction guide](../compaction.md) for the public API and local benchmark command.
