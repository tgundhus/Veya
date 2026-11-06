---
title: Single-pass score and boundary repair
description: A score-only adapter, independent evidence contrasts, and qualification with the existing gates.
---

# Single-pass score and boundary repair

The [choice-only routing repair](routing-repair-2026-11-06.md) improved the
original routing rubric and held the previous scores exactly. Five of its six
fresh score errors were level-zero cases. They involved routine work,
historical and simulated outages, or approval records for another item. The
last miss involved two valid approvals on the item in scope.

This checkpoint adds a width-64 score adapter to each decision-head layer.
The encoder and decision head run once per request under the tested canonical
policy. The score adapter applies only to score rows. The previous encoder,
decision head, choice adapter and binary and choice temperatures are retained.

## Training and holdout

The training bank covers live service impact, exact item and revision checks,
and numeric cutoffs. Hard targets use an integer level. Ambiguous targets use
0.5 or 1.5, which puts equal probability on the two adjacent levels through the
existing training encoder. These examples include conflicting records and
reports that leave both impact scope and fallback capacity unresolved.

Training uses four prior training banks for rehearsal. The published routing,
evidence, level-zero and older-control score cases are seen regression
training. The original rubric and earlier checkpoint acceptance corpus are
development fixtures. Separate calibration families fit the score
temperature. The trainer records the input roles and fingerprints and ranks
serialized weights using development, seen and calibration results.
The frozen fit used 219 new training rows, 607 rehearsal rows, 75 seen
regression rows and 40 calibration rows. It selected epoch 32 and a score
temperature of 1.675. Calibration hard scores were within tolerance on 18
of 29 cases.

The new holdout contains 72 cases from ten families. It has 30 labelled scores,
12 labelled choices, 10 labelled binary answers and 20 unlabelled candidate
boundaries. Review-agent checked the rubrics and cross-split overlap before it
was sealed in `666dc6a1`. The file SHA256 at evaluation was
`15fad48105f2fd9e8575d24f7ac9cc92fd923543f96207712226378ccf96b7d6`.
This is the raw hash of the Windows checkout with CRLF line endings.

The strict runner uses the existing 80% gates for each labelled question type,
the ±0.25 score tolerance, zero label flips or accuracy drop across profiles,
probability, confidence and score drift of at most 0.001, and at least one
measured top-two margin of at most 0.05. The fresh and original candidate
matrices use 32 CPU profiles across eager Torch and ONNX. The earlier fixtures
and fresh base use a serial profile on each backend.

## Qualification

The candidate remains experimental. Fresh ordinal accuracy stayed at 18/30.
Five level-zero, nine level-one, and four level-two scores were within
tolerance. Mean absolute error rose from 0.3730 to 0.3843. The 8/12 choice
accuracy was inherited from the routing checkpoint on these new families.
Binary answers held at 9/10.

The runner counted three near ties, all on labelled cases. The closest
unlabelled case was `scorefit-fresh-broadcast-console-8` at 0.1130. It has
a signed noise-floor pass with a missing serial that could name either
console, and no mapping result for the replacement. The base had one measured
unlabelled margin of 0.0405 for a channel mapping record that also lacked a
serial. The candidate therefore lost genuine boundary coverage even though
the fresh run met the runner's minimum boundary count.

| Corpus | Choice | Binary | Base ordinal | Candidate ordinal | Candidate boundaries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh acceptance | 8/12 | 9/10 | 18/30 | 18/30 | 3 |
| Original rubric | 4/5 | 5/5 | 4/5 | 4/5 | 0 |
| Earlier checkpoint acceptance | 9/12 | 12/12 | 11/12 | 9/12 | 1 |
| Older controls (seen scores) | 12/15 | 14/15 | 10/15 | 15/15 | 1 |
| Level-zero acceptance (seen scores) | 12/12 | 11/12 | 16/21 | 21/21 | 0 |
| Evidence acceptance (seen scores) | 12/12 | 12/12 | 11/21 | 21/21 | 3 |
| Routing acceptance (seen scores) | 16/18 | 7/8 | 12/18 | 18/18 | 0 |

The original matrix passed its accuracy and stability gates. Its boundary
gate failed. Its one missed score, `priority-counterfactual`, moved from
0.5914 to 0.4310 against level 1. The older controls retained the inherited
cross-backend choice confidence difference of `0.0010000000000000009`,
which exceeds the existing threshold. The earlier checkpoint acceptance
misses both its choice and score gates. The evidence acceptance fixture
passes, with two unlabelled near ties, on score cases used for training.
The prior routing and level-zero acceptance fixtures miss the boundary gate.
Across all seven comparisons, 1,440 complete choice and binary observations
matched the routing checkpoint exactly. The comparison uses the matching base
profile when available and the same backend's serial profile otherwise.

After opening the holdout, a development trial added 848 generated rehearsal
rows across 96 numeric and exact-item families. The generator varies item IDs
and includes distractors above and below the verified count at every level.
The selected trial reached 21/30 and produced two near ties on predeclared
unlabelled cases in the now-seen set. The required accuracy is 24/30, so this
trial did not advance to a second holdout. Blending the adapter weights
offline did not improve on 21/30.

## Runtime verification

The verifier matched all 218 base tensors exactly and found 12 added score
tensors. Both Torch and ONNX used one forward per mixed request, with exact
choice and binary outputs. Classic MCP and Code Mode each replayed two
retained decisions without loading a model, then recorded a fresh miss and
a repeat hit.

The warmed comparison measured 27 single forwards per model and backend,
with alternating order on four CPU threads. Another qualification job ran on
the same workstation during this sample.

| Backend | Base p50 | Candidate p50 | Base p95 | Candidate p95 | Median paired ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager CPU | 289.3 ms | 275.8 ms | 350.4 ms | 354.8 ms | 0.979× |
| Native ONNX CPU | 156.4 ms | 156.8 ms | 173.2 ms | 171.2 ms | 1.030× |

The split TypeScript export matched the full Torch model at sequence lengths
16 and 160. Maximum logit drift was 0.000063 and action probability drift was
zero. The optional TileLang CUDA graph path received source review and its
test skipped locally.

The shared runner passed all 122 suites. Pytest reported 1,635 tests and
4,238 subtests passed, with one skip. Ruff and byte compilation passed.

Raw matrices, training, protection, wide development and verification records
are in `research/results/` with the prefix `score_boundary_` and suffix
`_windows_20261106.json`.

## Reproduce

```bash
base=models/english-routing-repair-20261106
candidate=models/english-score-boundary-r3-20261106
fresh=research/evals/score-boundary-acceptance-v1.json
python research/scripts/train_score_boundary.py --base "$base" --out "$candidate" \
  --bank research/evals/score-boundary-training-v1.json \
  --rehearsal research/evals/checkpoint-repair-training-v1.json \
    research/evals/level-zero-training-v1.json research/evals/level-zero-contrast-v1.json \
    research/evals/evidence-repair-contrasts-v1.json \
  --regression research/evals/routing-repair-acceptance-v1.json \
    research/evals/evidence-repair-acceptance-v1.json research/evals/level-zero-acceptance-v1.json \
    research/evals/independent-repair-controls-v1.json \
  --development research/evals/qualification-v1.json research/evals/checkpoint-repair-acceptance-v1.json \
  --epochs 32
python -X utf8 scripts/export_onnx.py --model "$candidate" --output "$candidate/english-score-boundary.onnx"
python research/scripts/qualify_local.py --checkpoint "$candidate" \
  --onnx "$candidate/english-score-boundary.onnx" --decision-policy canonical \
  --corpus "$fresh" --out acceptance.json
python research/scripts/verify_checkpoint_repair.py --base "$base" --base-onnx "$base/english-routing.onnx" \
  --candidate "$candidate" --onnx "$candidate/english-score-boundary.onnx" \
  --corpus "$fresh" --score-only --out verification.json
```

For the wider development trial, generate a new rehearsal file with
`python research/scripts/expand_score_contrasts.py rehearsal.json`. Add
`rehearsal.json` to the rehearsal inputs, add the opened holdout to the
development inputs, choose a new output directory and set `--epochs 24`.
