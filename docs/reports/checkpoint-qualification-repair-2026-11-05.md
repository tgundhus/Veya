---
title: English checkpoint qualification investigation
description: Reproduced choice errors, measured checkpoint priors, an optional local correction and strict reruns without relaxing qualification gates.
---

# English checkpoint qualification investigation

The [original qualification](production-capabilities-2026-11-05.md) measured
60% on five labelled choice questions. This was the same three correct answers
in every eager CPU and ONNX execution profile. It was not an estimate of the
checkpoint's general accuracy. The two choice errors, one binary error and one
ordinal score outside tolerance were:

| Case | Rubric | Canonical output |
|---|---|---|
| `billing-evidence-removed` | Other: no request described | Billing 0.4350, other 0.3908, technical 0.1742 |
| `billing-counterfactual` | Technical: app crashes; billing is explicitly denied | Billing 0.5495, technical 0.3584, other 0.0920 |
| `authorization-evidence-removed` | False: no reply means no approval | True probability 0.5406 |
| `priority-evidence-removed` | Level zero: no impact or deadline | Expected level 0.3663, outside the ±0.25 tolerance |

The input path kept every instruction, option and state token for all 18 cases.
The grader uses the returned labels and the specified score tolerance. It does
not reorder choices or alter questions. Positive temperature scaling cannot
change the choice argmax, and the checkpoint's warning for 11-or-more-option
questions did not apply to these inputs. We found no decoding, truncation or
backend defect that explains the errors.

The checkpoint gave the missing-evidence billing case to billing under all six
choice orders. Only one order selected technical in the denial case, and the
mean over all six still selected billing. Picking that order using the gold
label would hide the problem. Fixed order averaging alone did not fix it.

Content-free inputs exposed a preference before state evidence: the ordinary
ordering averaged 0.5898 for billing and 0.6632 for a true authorization answer.
The score question assigned about 0.0210 to low priority and 0.6885 to medium.
These measurements support a question-dependent prior as a contributor to the
errors. They do not establish a general explanation for every negation failure.

Cyclic averaging alone left choice at 3/5 and noul at 4/5. Prior correction
alone fixed those two metrics, but still put one ordinal score outside tolerance.
The combination fixed all four original rubric errors. The
[raw diagnostic](https://github.com/tgundhus/Veya/blob/main/research/results/checkpoint_diagnosis_windows_20261105.json)
retains token sequences, original logits, all six choice orders, fixed probes and
the separate production-policy run. This is diagnostic evidence, separate from
the strict qualification matrix.

## Correction and validation protocol

The [optional decision policy](../decision-policy.md) combines cyclic order
averaging and fixed content-free prior correction in both actual inference
backends. This changes inference policy; the checkpoint weights, default path,
questions, labels and gates remain unchanged. Canonical and corrected decisions
cannot share a cache entry. Corrected confidence is explicitly uncalibrated.

We selected this correction after examining the original errors, so the
original fixture is a regression set. The policy has no fitted parameters, and we did not choose
an option order or tune a threshold against the answers. An independent author
created additional synthetic controls without reading the original corpus,
diagnosis or outputs. Their cases were sealed before the corrected policy ran:
45 labelled decisions (15 each for choice, noul and score) and three unlabelled
mixed or conditional cases, all in distinct source families. The control file's
SHA256 is `936c4911bf9b5ea61f95803daadef911a7c45edce7551500f92118773977baa1`.
These controls are a check for regressions on unseen prompts; they are not an
expert-adjudicated production dataset.

The original matrix is still two required CPU cells, two repetitions, batch
sizes one and four, and concurrency one and four. It still requires 0.80 for
each labelled accuracy metric, zero label flips, at most 0.001 probability and
score drift, and at least one actual probability gap of at most 0.05. Coverage
uses the probabilities the selected policy returns. A more accurate prediction
may move away from the boundary; it cannot inherit the old case's margin.

## Original fixture rerun

All 32 profiles completed. The two backends returned the same labels, scores,
probabilities, confidence, action values and operational flags in every profile.
All 15 labelled answers meet their original rubrics:

| Metric | Canonical | Context balanced |
|---|---:|---:|
| Choice accuracy | 3/5 (60%) | 5/5 (100%) |
| Noul accuracy | 4/5 (80%) | 5/5 (100%) |
| Scores within ±0.25 | 4/5 (80%) | 5/5 (100%) |
| Score mean absolute error | 0.08264 | 0.04938 |
| Measured boundary cases | 1 | 0 |

The corrected missing-evidence routing case picks other at 0.5382. The explicit
denial picks technical at 0.5152. The missing authorization falls to a true
probability of 0.4214, and missing outage evidence yields score 0.0440.

**The strict verdict is still a failure.** The only remaining gate failure is
`reference has 0 measured boundary cases; need 1`. The closest returned margin
is 0.1572. Both cells meet their per-profile gates, but the matrix keeps
`passed: false` and `production_qualified: false`. No artificial ties or
substituted canonical probabilities were used to count coverage.

Fresh predictions cost more. In the first serial profile, eager p95 moved from
285 ms to 1,989 ms and ONNX p95 from 205 ms to 1,781 ms, across separate local
runs on the same artifacts and thread settings. Each three-option single
prediction executes three real-state rows plus nine probe rows. These samples
include operator warmup, have only 18 calls per cell, and do not certify a latency
SLO. Retained decisions can replay without this work.

[Raw original-fixture rerun](https://github.com/tgundhus/Veya/blob/main/research/results/qualification_context_balanced_windows_20261105.json).

## Independent controls

Each policy ran the same 48 sealed inputs once serially on each backend. This
separate check used the same gate thresholds and measured boundary definition;
the original fixture remains the batch, ordering and concurrency stress test.

| Metric | Canonical | Context balanced |
|---|---:|---:|
| Choice accuracy | 9/15 (60.0%) | 11/15 (73.3%) |
| Noul accuracy | 13/15 (86.7%) | 12/15 (80.0%) |
| Scores within ±0.25 | 8/15 (53.3%) | 6/15 (40.0%) |
| Score mean absolute error | 0.40412 | 0.44773 |
| Measured boundary cases | 5 | 5 |

Both policies fail the 80% choice and score gates on these controls. The
correction fixed three choice errors and introduced one, fixed one noul error
and introduced two, and added two scores outside tolerance. No label,
probability, score or action drift was observed between backends within a
policy; maximum confidence drift was 0.0001, within the unchanged 0.001 limit.
These results do not support changing the default or a broad model-quality claim.
The policy remains experimental. A stronger checkpoint would need separately
reviewed training and calibration data and another untouched acceptance set.

[Canonical control report](https://github.com/tgundhus/Veya/blob/main/research/results/qualification_controls_canonical_windows_20261105.json)
and [corrected control report](https://github.com/tgundhus/Veya/blob/main/research/results/qualification_controls_balanced_windows_20261105.json).

## Review and verification

The inference policy and its service/cache contracts are in `18a53337`.
Qualification integration is in `f9048a82`; the deployment and existing batch
contract checks are aligned in `703a9854`. Independent review-agent passes
covered inference, cache/replay identity, resource accounting, qualification,
packaging, the diagnostic and the final evidence. Reviewers recomputed the
labelled metrics, drift, quantiles and 249 source digests across four artifacts.

The 122 shared suite files passed after targeted reruns of three script checks
affected by the batch refactor and the Nix policy option. The pytest group had
1,628 passes, 4,238 subtests and one TileLang dependency skip. The hook/API
contract passed 908 checks. Ruff and compilation passed. The TypeScript HTTP
client passed 34 tests; the Python HTTP client wheel and rebuilt sdist each ran
68 tests with one platform skip. The standalone ONNX wheel and outside-checkout
source rebuild passed their installed consumers, including corrected inference,
Code Mode budget enforcement and model-free replay. All 36 bundled module
digests matched the tested sources. Strict documentation checks passed.

## Reproduce locally

```bash
python research/scripts/diagnose_checkpoint.py --checkpoint /models/laya \
  --out checkpoint-diagnosis.json
python research/scripts/qualify_local.py --checkpoint /models/laya \
  --onnx /models/english.onnx --decision-policy context-balanced-v1 \
  --out qualification-context-balanced.json
```

Both commands use local artifacts. The qualification command reports a nonzero
exit when any configured gate fails, including missing measured boundary cases.
The historical canonical report remains available for comparison. A reviewed
baseline for automated drift gating must match the new inference policy.

For the independent control reports, use `--corpus research/evals/independent-repair-controls-v1.json`
with `--repeats 1 --batch-sizes 1 --concurrencies 1`; run once with each explicit
`--decision-policy` value and separate output files.
