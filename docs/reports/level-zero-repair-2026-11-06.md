---
title: English level-zero score repair
description: Single-pass score-head training, sealed boundary coverage and unchanged qualification gates.
---

# English level-zero score repair

The [previous checkpoint repair](checkpoint-repair-2026-11-06.md) missed the
fresh ordinal gate because it overestimated four cases whose expected level
was zero. A routine request with a working service could score as an active
disruption. Older controls also showed that plans and records for other items
could be mistaken for completed work. This follow-up starts with that repaired
English checkpoint and trains its existing decision head.

## Data and selection

Two supplements provide 121 score examples in 18 families, including 54
level-zero examples. Impact cases distinguish routine work, historical or
simulated failures, active failures with a usable alternative, and urgent
loss of service. Completion cases distinguish planned work, evidence for a
different item, and completed steps. The 18 score calibration examples cover
two other families. The previous training bank supplies 138 rehearsal examples.

Training keeps the encoder and action head weights frozen. The selected
recipe uses the presented option order, penalizes errors in the expected
semantic level, and gives extra weight to level-zero mistakes. Choice and
binary rehearsal retains most of the prior model's distribution. Calibration
minimizes squared expected score error on the separate calibration split. The
recipe retains the base temperatures for three-option choice and binary
decisions. Epoch selection uses the original rubric, older controls, and the
previous acceptance corpus as development data.

The rotated head trial, a trial that updated the last four encoder layers,
and tests of heavier zero weighting, weight blending, and more generated
examples failed to improve the selected development result. The wider
hand-labelled head trial selected epoch 8. It fixed one of the four prior
level-zero misses while giving up one higher-level hit. Its older
control score fell from 9/15 to 8/15. A second run from the tracked recipe
reproduced the selected weight digest exactly before qualification.

The temperature sweep on the first rotated trial showed two causes. A lower
temperature brought the original priority score within tolerance. The four
previously held-out level-zero misses still ranked a higher level first
across the temperature sweep. Positive temperature scaling preserves that
ranking, so a temperature change alone cannot recover those cases.

The new acceptance corpus contains 45 labelled cases and 21 predeclared
unlabelled boundary candidates in seven families. Among the 21 labelled score
cases, nine expect level zero, six level one, and six level two. The boundary
candidates contain competing requests, incomplete evidence, or conflicting
reports. They count toward measured coverage only if the model returns a
top-two gap within the configured limit. The corpus was sealed in `2ddece43`,
before any acceptance predictions. Its SHA256 is
`095910b9672fc5fd8cfbcd3497120887a50c119747aef6f11e384d00a237f085`.

The unchanged strict gates require 80% accuracy per type, score tolerance
of ±0.25, zero label flips, probability and score drift at most 0.001, and at
least one measured top-two gap of at most 0.05. The strict CPU matrix covers
Torch and ONNX with two repeats, batch sizes one and four, and concurrency
one and four.

## Fresh qualification

**The candidate failed qualification.** The same accuracy counts held in
all 32 candidate profiles. The prior checkpoint comparison used a serial
profile on each backend with the same thresholds.

| Fresh acceptance metric | Prior repair | Level-zero candidate |
| --- | ---: | ---: |
| Choice accuracy | 9/12 | 9/12 |
| Binary accuracy | 10/12 | 10/12 |
| All scores within ±0.25 | 10/21 | 8/21 |
| Level zero within ±0.25 | 2/9 | 4/9 |
| Level one within ±0.25 | 4/6 | 2/6 |
| Level two within ±0.25 | 4/6 | 2/6 |
| Score mean absolute error, Torch reference | 0.47585 | 0.49191 |

The candidate brought two routine seismograph reports within tolerance. The
contact-list update moved from 1.1079 to 0.2366, and the simulated outage moved
from 0.8185 to 0.0256. Four previously correct higher-level scores then fell
below tolerance. For example, the active network-wide detection loss moved
from 1.7903 to 0.4337 against an expected score of 2. The score and choice
accuracy gates both failed.

Two of the 21 unlabelled cases met the actual boundary gate. In
`stone-sample-clearance-9`, a signed ownership check and an unattributed lab
result leave a 0.0379 gap between the two highest probabilities. In
`stone-sample-clearance-10`, the completed checks might cover a different
container, and the gap is 0.0145. Their full probability vectors and scores
matched in every profile. Across the entire fresh matrix, there were zero
label or operational flag flips. Maximum probability, score, and confidence
drift was 0.0001, and action drift was zero.

## Earlier fixtures

The original rubric completed 32 profiles. The candidate corrected the
billing denial routing decision and met every accuracy gate with 5/5 choice,
5/5 binary, and 4/5 scores within tolerance. The priority case without any
reported impact moved from 0.5629 to 0.3160 and still missed tolerance. The
original fixture's only strict failure was the absence of a measured near-tie.

On the previous acceptance set, the candidate scored 10/12 choice, 10/12
binary, and 8/12 ordinal in both serial profiles. It corrected the greenhouse
label-ordering request at level zero and missed the observatory's level-one
fault with a usable replacement telescope. The net score count was the same
as the prior checkpoint and below the score gate. Both measured boundaries
were labelled cases.

The older controls scored 12/15 choice, 14/15 binary, and 8/15 ordinal in
both serial profiles, compared with 14/15, 14/15, and 9/15 for the prior
checkpoint. All five level-one scores met tolerance, but only one of five
level-two scores did. Four measured boundaries included one unlabelled
case. The score gate failed again, and the candidate remains experimental.

## Single-pass performance

The paired benchmark warmed nine prompts and measured each one three times
with alternating order and four CPU threads. Both checkpoints made 27
measured forwards per backend, one per request. All tensor shapes and dtypes
matched. The encoder and action head stayed identical, and the 31 updated
tensors belonged to the existing head, scorer, and type embeddings.

| Backend | Prior p50 | Candidate p50 | Prior p95 | Candidate p95 | Paired median ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager CPU | 163.8 ms | 161.9 ms | 177.4 ms | 177.2 ms | 0.997× |
| Native ONNX CPU | 148.5 ms | 151.2 ms | 173.8 ms | 173.5 ms | 1.013× |

Both local MCP modes returned two retained predictions before loading the
checkpoint, then recorded one fresh miss and one repeat hit. The comparison
uses a small warmed workstation sample. It gives the observed per-request
cost of this checkpoint and does not establish a latency SLO.

## Review and evidence

Review-agent checked the rubrics, split isolation, ordinal loss, device
placement, score calibration, and qualification evidence. The trainer rejects
duplicate banks by content, including aliases and copies. The candidate
artifact was trained before that input guard, with two distinct score banks.
Its original trainer source is recorded in `05f897e9`. The 322 focused
qualification tests, Ruff, compilation, and the strict documentation and
site checks passed.

Raw records under `research/results/` use the `level_zero_` prefix and
`_windows_20261106.json` suffix. They include `development_trials`, `training`,
`verification`, `acceptance_base`, `acceptance`, `original`,
`previous_acceptance`, and `older_controls`.

## Reproduce

The recorded training run used Torch 2.11.0+cu128 and four CPU threads with
training on CUDA. Qualification and paired measurements used the CPU runtime.

```bash
python research/scripts/train_checkpoint_repair.py --base models/english-repair-20261106 \
  --level-zero-data research/evals/level-zero-training-v1.json research/evals/level-zero-contrast-v1.json \
  --level-zero-canonical --device cuda --threads 4 --epochs 24 \
  --development research/evals/qualification-v1.json research/evals/independent-repair-controls-v1.json \
  research/evals/checkpoint-repair-acceptance-v1.json --output models/english-level-zero-wide-20261106
python -X utf8 scripts/export_onnx.py --model models/english-level-zero-wide-20261106 \
  --output models/english-level-zero-wide-20261106/english-level-zero.onnx
python research/scripts/qualify_local.py --checkpoint models/english-level-zero-wide-20261106 \
  --onnx models/english-level-zero-wide-20261106/english-level-zero.onnx --decision-policy canonical \
  --corpus research/evals/level-zero-acceptance-v1.json --out acceptance.json
python research/scripts/verify_checkpoint_repair.py --base models/english-repair-20261106 \
  --base-onnx models/english-repair-20261106/english-repair.onnx --candidate models/english-level-zero-wide-20261106 \
  --onnx models/english-level-zero-wide-20261106/english-level-zero.onnx \
  --corpus research/evals/level-zero-acceptance-v1.json --out verification.json
```
