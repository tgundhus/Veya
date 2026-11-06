---
title: Choice-only routing checkpoint repair
description: Single-pass choice adapters, protected score outputs and fresh qualification.
---

# Choice-only routing checkpoint repair

The [evidence-grounding follow-up](evidence-repair-2026-11-06.md) improved
ordinal scoring but lost three fresh choice hits. This repair starts from
that checkpoint and learns a small correction for choice questions. The
checkpoint adds a width-64 adapter before each of its two decision-head
layers. The existing encoder runs once. A question-type mask applies each
correction to choice rows, including requests that contain all three types.
The checkpoint retains every original tensor and the score and binary
temperatures. The repair is opt-in through the checkpoint configuration.

## Training and acceptance

The new local bank provides 64 labelled routing examples in eight families
and 18 calibration examples in two other families. The examples distinguish
active requests from settled issues, general contact, and references supplied
before the requester describes an action. The trainer also uses earlier local
routing banks and 2,800 examples made from CLINC's `train` and `oos_train` splits.
The [CLINC release](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de)
contains the source intent labels and out-of-scope examples.

The 12 published choice labels from the evidence checkpoint's failed
acceptance run become seen regression training. The four older fixtures and
the seen regression cases inform epoch and strength selection. Choice adapters
are the only trainable parameters. Training emphasizes examples the base
checkpoint gets wrong and retains some of its probability distribution on
correct examples. The selected epoch was 12, with a 0.75 correction strength
folded into the saved adapter weights. Choice temperature uses the separate
calibration families and existing calibration rows.

The fresh acceptance corpus has 63 cases in eight families, including 44
labelled cases and 19 predeclared unlabelled cases. It was sealed in `83ae3aa8`
before any predictions on it. Its SHA256 is
`df7396336b4363c4a2f912bfc74ebe7cc99d1749a4d9f925062a703dba0a2b80`.
Review-agent checked the rubrics, competing evidence and split isolation.

The strict fresh and original matrices use the existing thresholds and 32
Torch and ONNX profiles per corpus. They require 80% accuracy for each type,
score tolerance of ±0.25, zero label flips or accuracy drop across profiles,
probability, confidence, and score drift of at most 0.001, and at least one
measured top-two gap of at most 0.05. The older and seen regression fixtures
use the same thresholds in serial profiles.

## Qualification

The candidate clears the routing threshold on the fresh and original
matrices. It remains experimental. Fresh scores miss the existing accuracy
gate, and neither matrix has a measured boundary at the existing
0.05 cutoff. The original matrix meets all its accuracy and stability gates.

| Corpus | Base choice | Candidate choice | Binary unchanged | Scores within ±0.25 unchanged | Candidate boundaries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh acceptance | 17/18 | 16/18 | 7/8 | 12/18 | 0 |
| Original rubric | 3/5 | 4/5 | 5/5 | 4/5 | 0 |
| Prior checkpoint acceptance | 10/12 | 9/12 | 12/12 | 11/12 | 2 |
| Older controls | 13/15 | 12/15 | 14/15 | 10/15 | 2 |
| Prior level-zero acceptance | 11/12 | 12/12 | 11/12 | 16/21 | 0 |
| Seen routing regression | 5/12 | 12/12 | 12/12 | 11/21 | 3 |

On the fresh set, the candidate fixed `library-lock-helpdesk-5`, which asks
for reception's number and says there is no lost card or broken door. It lost
two credential requests, `library-lock-helpdesk-1` and `library-lock-helpdesk-2`.
The score breakdown within tolerance was 1/6, 6/6, and 5/6 at levels 0, 1, and 2.
The nearest fresh choice case, `art-cover-service-desk-8`, had a gap of 0.0801.
The base checkpoint had two labelled gaps within 0.05; neither remained there.
The 19 unlabelled candidates were declared before evaluation and kept as-is.

On the original matrix, the adapter fixed `billing-counterfactual` and still
missed `billing-evidence-removed`. Across the two matrices there were zero
label or flag flips between profiles; the maximum returned score, probability,
and confidence drifts were at most 0.0001.

On the prior checkpoint acceptance fixture, the candidate fixed one routing
case and lost two, falling below the choice gate. It retained both measured
boundaries, including an unlabelled gap of 0.0115. The older controls still
miss the score gate. Their cross-backend confidence drift was recorded as
`0.0010000000000000009`, just above the unchanged `0.001` gate. The prior
level-zero acceptance fixture misses its score and boundary gates.

The candidate fixed all seven prior routing misses on the seen regression.
Those 12 labels were training data. On that fixture the earlier score gain
from 7/21 to 11/21 was preserved, including 4/9 at level zero. The score gate
still fails. Its unlabelled `canteen-allergen-release-7` still measured a
0.0368 gap. All six comparisons
matched 1,960 score and binary observations exactly against the base,
including unlabelled observations. The comparison uses the corresponding
base profile when available and the same backend's serial profile otherwise.

## Single-pass verification

The frozen checkpoint preserved all 206 original tensors and added 12 adapter
tensors. The verifier also checked the encoder and tokenizer metadata and
the nonchoice configuration. Both Torch and ONNX used one forward for a
mixed choice, score, and binary request. The score and binary probabilities,
scores, confidence, action values, and labels matched exactly for that request.

The warmed benchmark measured nine prompts three times per checkpoint with
alternating order and four CPU threads. Each model made 27 measured forwards
per backend. Median latency stayed close to baseline, and both p95s rose in
this sample. The run is too small to establish a production tail latency bound.

| Backend | Base p50 | Candidate p50 | Base p95 | Candidate p95 | Median paired ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager CPU | 147.2 ms | 147.5 ms | 177.5 ms | 193.6 ms | 1.003× |
| Native ONNX CPU | 153.7 ms | 148.0 ms | 182.8 ms | 194.9 ms | 0.988× |

Classic MCP and Code Mode both served two retained decisions before loading
the checkpoint. Each observed one fresh miss and one repeat hit.

The TypeScript split exporter applies the same adapter step as the full
decision model. A real split export matched full-model reference logits
within 0.000012 at sequence lengths 16 and 160, with zero reported action
probability drift. For the optional TileLang CUDA graph path, adapter weight
casts run with autocast caching disabled so graph replay owns the casts. That
path was reviewed in source; the optional TileLang test skipped locally.

The local suite passed all 122 suites: pytest reported 1,634 tests passed,
one skipped, and 4,238 subtests passed. Ruff and byte compilation passed.

The raw records in `research/results/` use the prefix `routing_repair_` and
suffix `_windows_20261106.json`. They include `training`, `verification`,
`acceptance_base`, `acceptance`, `original`, `prior_acceptance`,
`older_controls`, `prior_level_zero`, `seen_regression`, and `protection`.

## Reproduce

The recorded training run used local Torch 2.11.0+cu128 and four CPU threads.
Obtain `data/data_full.json` and `data/domains.json` from the pinned CLINC
release above. The training record contains their hashes, the checkpoint
hashes and the exact trainer and runtime source hashes.
The frozen weights were produced with trainer `8e27710a`. The CUDA graph
guard and duplicate-input checks landed afterward. The final benchmark and
qualification used the guarded runtime.

```bash
banks=(research/evals/checkpoint-repair-training-v1.json
       research/evals/evidence-repair-contrasts-v1.json research/evals/routing-repair-training-v1.json)
dev=(research/evals/qualification-v1.json research/evals/independent-repair-controls-v1.json
     research/evals/checkpoint-repair-acceptance-v1.json research/evals/level-zero-acceptance-v1.json)
python research/scripts/train_routing_repair.py --base models/english-evidence-repair-r2-20261106 \
  --out models/english-routing-repair-20261106 --clinc /data/clinc/data_full.json \
  --domains /data/clinc/domains.json --banks "${banks[@]}" --development "${dev[@]}" \
  --regression research/evals/evidence-repair-acceptance-v1.json --epochs 12
python -X utf8 scripts/export_onnx.py --model models/english-routing-repair-20261106 \
  --output models/english-routing-repair-20261106/english-routing.onnx
python research/scripts/qualify_local.py --checkpoint models/english-routing-repair-20261106 \
  --onnx models/english-routing-repair-20261106/english-routing.onnx --decision-policy canonical \
  --corpus research/evals/routing-repair-acceptance-v1.json --out acceptance.json
python research/scripts/verify_checkpoint_repair.py --base models/english-evidence-repair-r2-20261106 \
  --base-onnx models/english-evidence-repair-r2-20261106/english-evidence.onnx \
  --candidate models/english-routing-repair-20261106 \
  --onnx models/english-routing-repair-20261106/english-routing.onnx --routing-only \
  --corpus research/evals/routing-repair-acceptance-v1.json --out verification.json
```
