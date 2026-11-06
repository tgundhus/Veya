---
title: Ordinal continuation from the wider trial
description: Continued local score training and a fresh audit of the plausible levels behind near ties.
---

# Ordinal continuation from the wider trial

The [wider score trial](score-boundary-repair-2026-11-06.md) reached 21/30 on
the then-open score fixture. The new training starts from that checkpoint.
The wider trial still missed full outages, completed checks on an exact item,
and high counts. The continuation fits only the existing score adapter
weights and the score temperature.

## Training and holdout

The new local bank has 196 training cases in 22 families and 54 calibration
cases in six other families. It covers exact evidence identifiers, live
scope and alternatives, and verified counts at both cutoffs. Numeric
calibration uses different sentence frames from training. The 1,067
rehearsal cases come from the previous bank and the wider trial's generated
bank. The five published score fixtures supply 105 seen regression cases,
including the 30 labelled scores from the prior holdout. The original
rubric and earlier checkpoint acceptance fixture inform selection.

Training adds two rotated option orders per local or seen case. It keeps
the targets and the numeric level values aligned with those orders. The
loss anchors already correct cases to the wider checkpoint and includes a
penalty for imbalance between the supported levels on exact midpoint
targets. Calibration ranks temperatures by hard scores within tolerance
and uses the calibration loss to break ties.

The combined calibration and development ranking selected epoch 8, with
0.85 of the learned update and a score temperature of 0.5. That variant
hit 29 of 38 hard calibration cases.
A short follow-up emphasized midpoint training and counted correct adjacent
near ties alongside hard calibration hits. Two of its variants reached two
correct calibration near ties, with 24/38 and 23/38 hard hits. The follow-up
did not advance to the fresh holdout.

Review-agent checked the 79 fresh cases for rubric consistency and family
overlap before they were sealed in `7a62f01d`. They include 36 labelled
scores, 15 choices, 10 binary answers and 18 unlabelled ordinal cases. Each
unlabelled case declares the adjacent levels its evidence supports. The raw
Windows checkout SHA256 is
`3ef1dd482115e5d896bdc2012f8b512d7f8b79d681436ff0c11fb22896b9aac1`.

## Qualification

The strict runner used the existing 80% accuracy gates, the ±0.25 score
tolerance, zero label flips or accuracy drop across profiles, probability,
confidence and score drift at most 0.001, and at least one measured margin
at or below 0.05. Fresh and original candidate runs each use 32 Torch and
ONNX CPU profiles. Earlier fixtures and wider baselines use one serial
profile per backend.

The candidate remains experimental. Fresh score accuracy improved from
16/36 to 20/36, below the 29/36 gate. Mean absolute error rose from 0.5338 to
0.5535 as some remaining misses got larger. Level-zero hits stayed at
5/12, level-one hits moved from 5/12 to 7/12, and level-two hits moved
from 6/12 to 8/12. The wider checkpoint and candidate both scored 10/15
in choice and 9/10 in binary. The fresh choice gate also fails.

The fresh qualification runner counted one near tie on a labelled routing
case that the wider checkpoint already had. The semantic audit counts an
unlabelled near tie only when its unique top-two set matches the two declared
levels in every profile. The wider checkpoint had one such case in its
serial profiles, with a gap of 0.0427 on `ordinal-fresh-heat-exposure-7`.
The candidate had none. Its closest case with the correct pair was
`ordinal-fresh-waterjet-workshop-7` at 0.3320. Across the fresh matrix
there were zero label or operational flag flips and maximum probability,
confidence and score drift were each at most 0.0001.

| Fixture | Wider score hits | Continued score hits | Candidate choice | Candidate binary |
| --- | ---: | ---: | ---: | ---: |
| Fresh acceptance | 16/36 | 20/36 | 10/15 | 9/10 |
| Original rubric | 5/5 | 5/5 | 4/5 | 5/5 |
| Prior score acceptance, now seen | 21/30 | 28/30 | 8/12 | 9/10 |

On the original rubric, mean absolute score error improved from 0.0654
to 0.0205. The matrix passed its accuracy and stability gates, with zero
measured boundaries. On the prior score fixture, six more level-two
answers and one more level-one answer entered tolerance in aggregate.
The fixture's choice gate fails and it lost its measured boundaries.

The candidate had 10/12 score hits on earlier independent acceptance,
where its 9/12 routing result misses the gate. It reached every labelled
score on the older controls, level-zero, evidence and routing fixtures.
Those scores were seen in training. The evidence fixture passes. The
level-zero and routing fixtures miss boundary coverage. Older controls
still have the inherited cross-backend confidence difference of
`0.0010000000000000009`.

## Runtime verification

The verifier matched 218 tensors exactly and found 12 updates, all in the
existing score adapters. Both Torch and ONNX used one forward per mixed
request, with exact choice and binary outputs. Classic MCP and Code Mode
each replayed two retained decisions without loading the model, then
recorded one fresh miss and one repeat hit.

Across eight report comparisons, 1,568 of 1,580 protected observations
matched exactly. The remaining 12 were the same choice case in batched
Torch profiles versus the wider serial baseline, with probability drift
of 0.0001 and no label, confidence or action change. Fresh serial profiles
matched exactly on both backends. The comparisons use the wider checkpoint
for the fresh, original and prior score fixtures, and the earlier
score candidate for the five historical fixtures.

The warmed comparison made 27 measured calls per model and backend. The
median of the paired ratios was 1.026× in Torch and 0.945× in ONNX. The
fresh matrix was running on the same workstation during these samples.

The shared runner passed all 122 suites. Pytest recorded 1,640 passed,
4,238 subtests passed and one skipped. The six focused verification and
boundary-audit checks passed after the final audit change. Ruff and byte
compilation passed.

Raw evidence is in `research/results/` with prefix `ordinal_continuation_`
and suffix `_windows_20261106.json`. It includes the saved matrix, paired
observations, semantic boundary audit, fit records and runtime verification.

## Reproduce

This reproduction starts from the pinned R5 wider checkpoint. Its
`model.safetensors` SHA256 is
`0ddee2ab8c799e418a0bbddba90b7e4f50472edc1720c33443b4e549efb47645`.
The earlier `score_boundary_wide_development_windows_20261106.json` record
has the R5 trainer and input fingerprints, split roles and selected epoch.
Generate the same standalone rehearsal bank with
`python research/scripts/expand_score_contrasts.py rehearsal.json`.

Run the training command from a checkout at `aeae409f` to match the
recorded trainer source. Run export, qualification, audit and verification
from the current branch using the same local checkpoint directories.

```bash
base=models/english-score-boundary-r5-20261106
candidate=models/english-ordinal-continuation-r2-20261106
fresh=research/evals/ordinal-continuation-acceptance-v1.json
python research/scripts/train_score_boundary.py --base "$base" --out "$candidate" \
  --bank research/evals/ordinal-continuation-training-v1.json \
  --rehearsal research/evals/score-boundary-training-v1.json rehearsal.json \
  --regression research/evals/routing-repair-acceptance-v1.json \
    research/evals/evidence-repair-acceptance-v1.json research/evals/level-zero-acceptance-v1.json \
    research/evals/independent-repair-controls-v1.json research/evals/score-boundary-acceptance-v1.json \
  --development research/evals/qualification-v1.json research/evals/checkpoint-repair-acceptance-v1.json \
  --epochs 10
python -X utf8 scripts/export_onnx.py --model "$base" --output "$base/english-wide.onnx"
python -X utf8 scripts/export_onnx.py --model "$candidate" --output "$candidate/english-ordinal.onnx"
python research/scripts/qualify_local.py --checkpoint "$candidate" \
  --onnx "$candidate/english-ordinal.onnx" --decision-policy canonical \
  --corpus "$fresh" --out acceptance.json
python research/scripts/check_ordinal_boundaries.py --corpus "$fresh" \
  --report acceptance.json --out boundaries.json
python research/scripts/verify_checkpoint_repair.py --base "$base" --base-onnx "$base/english-wide.onnx" \
  --candidate "$candidate" --onnx "$candidate/english-ordinal.onnx" \
  --corpus "$fresh" --score-only --out verification.json
```
