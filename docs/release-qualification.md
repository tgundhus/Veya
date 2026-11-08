---
title: Release qualification
description: Qualify local decision backends, precision, batching and queue behaviour against fixed source families and reviewed release observations.
---

# Release qualification

`veya-evals qualify` checks a fixed decision corpus across local backend/precision
cells, repeated calls, batch sizes, reversed input order and unrelated batch
neighbours. Its reference is the first serial observation. A saved reviewed
baseline also catches a code change that changes every current cell together.
All inference and artifact inspection run locally. Models are never downloaded.

Qualification checks agreement separately from correctness. Exact final-decision
retention still belongs to [required caching](consistency.md); these checks must
run without a decision cache so replay cannot hide a regression.

## Fixed cases and source families

The checked-in `research/evals/qualification-v1.json` is a versioned engineering
fixture: 22 cases in five source families, with 18 evaluation cases covering
`choice`, `noul` and ordinal `score`. Three ambiguous candidates deliberately have
no gold label. They are not claimed to produce 49/51 probabilities: the report
records how many cases actually fall within the configured probability margin.
The other cases have explicit synthetic rubric labels. Agreement with those
labels is a fixture check, not a general model-accuracy estimate.

Each case supplies `id`, `family`, `split`, `variant`, `state`, `questions` and
`expected`. Splits are `training`, `calibration` and `evaluation`. A complete
source family stays in one split; identical state/question inputs cannot occur
across different splits under another family name. Paraphrases, distractors,
removed evidence and counterfactuals therefore cannot leak across partitions.
Only evaluation cases are scored. Use your reviewed domain examples for an
actual release decision; record their source-family provenance honestly.

```bash
veya-evals qualify-validate research/evals/qualification-v1.json
```

`veya.qualification.split_families()` can assign complete families using a seeded
metadata-only hash. It does not select cases using predictions, confidence or
correctness. Inspect domain/type coverage after splitting. Family IDs cannot
automatically establish that two semantically related documents are independent.

## Define the local deployment matrix

```json
{
  "schema": "veya-qualification-matrix/1",
  "reference": "cpu-fp32",
  "batch_sizes": [1, 2, 4],
  "concurrencies": [1, 4],
  "repeats": 3,
  "limits": {
    "max_label_flip_rate": 0,
    "max_probability_drift": 0.001,
    "max_score_drift": 0.001,
    "max_accuracy_drop": 0,
    "min_accuracy": 0.9,
    "boundary_margin": 0.05,
    "min_boundary_cases": 10,
    "max_p95_ms": 500,
    "max_p99_ms": 750
  },
  "cells": [
    {
      "name": "cpu-fp32",
      "checkpoint": "/models/reviewed-english",
      "backend": "eager",
      "device": "cpu",
      "precision": "fp32",
      "required": true
    },
    {
      "name": "onnx-cpu",
      "checkpoint": "/models/reviewed-english",
      "onnx_path": "/models/exports/english.onnx",
      "backend": "onnx",
      "device": "cpu",
      "precision": "graph",
      "required": true
    }
  ]
}
```

The numbers above are examples, not recommended universal thresholds. Set them
from the errors, coverage and service budgets your application accepts. Paths are
resolved relative to the matrix file. A checkpoint must exist locally and contain
its configuration, tokenizer and weights (ONNX cells instead use their graph's
weights). ONNX external tensor files are hashed too. `precision: "graph"` identifies
the pinned graph; it does not infer or assert a graph-wide floating-point dtype.

An optional cell `decision_policy` selects `canonical` or `context-balanced-v1`.
Omit it to use the checkpoint's default. The runner verifies the effective policy,
records a noncanonical policy in the cell identity and checks returned calibration
status for drift. Use the same policy across backends in a release matrix. To
compare saved reports, use a reviewed baseline from that same policy. See
[context-balanced decisions](decision-policy.md) for execution and cache behavior.

Torch cells accept `eager`, `compile` and `tilelang`, with explicit CPU, CUDA or
MPS devices and compatible `fp32`, `fp16` or `bf16` precision. Unsupported hardware
or dependencies cannot substitute another cell. Required cells fail the gate;
recognized unavailability on an explicitly optional cell is reported as `skipped`.
Corrupt artifacts, arbitrary model errors and runtime identity changes always
fail, including for optional cells. Skipped targets are outside the qualified
scope. Hardware backends also need verifiable model-forward execution; a backend
that bypasses that witness fails rather than claiming an unobserved precision.

## Run and review the gate

```bash
veya-evals qualify cases.json --matrix matrix.json --json candidate.json
veya-evals qualify cases.json --matrix matrix.json --baseline reviewed.json \
  --json candidate.json
veya-evals qualify-compare candidate.json --baseline reviewed.json \
  --json comparison.json
```

Exit status is `0` for passing configured gates, `1` for a qualification failure,
and `2` for malformed inputs or invocation errors. A release run requires explicit
`min_accuracy` and at least one measured near-boundary case. `--diagnostic` allows
agreement-only exploration. A diagnostic pass without explicit accuracy and
boundary gates is never labelled `production_qualified`.

The qualification report includes every completed observation and local call
latency, measured boundary cases, drift/label flips, rubric metrics, actual
backend/device/precision, artifact and tokenizer digests, runtime versions, source
hashes and declared thresholds. The existing labelled evaluation harness computes
`choice_accuracy`, `noul_accuracy`, `score_mae` and `score_within_0.25`. The minimum
accuracy gate applies to every present accuracy/score-within metric. Score is a
continuous expected ordinal level, so score drift has its own gate; discrete label
flips apply to choice and noul answers.

`confidence`, `answer_confidence` and `action.act_probability` are checked for
finite values in `[0,1]`; confidence and action drift use the probability-drift
budget. Presence changes in these fields fail qualification. Returned
abstention/review flag values must agree at the default zero label-flip budget. Operational flag
changes have their own reported flip rate, using `max_label_flip_rate` as their
limit. A stable primary label cannot hide a changed downstream
review or abstention decision.

Local cells load one at a time. Torch forward hooks verify the device and active
autocast dtype for each actual forward, including fallback attempts; configuration
and calibration are checked before and after calls. Float32 matmul policy,
deterministic mode and relevant CPU/CUDA math flags are recorded and checked too;
autocast dtype alone does not identify every kernel's arithmetic. Saved release
comparisons require the same numerical policy. Recent PyTorch versions expose
[MKLDNN](https://docs.pytorch.org/docs/stable/notes/mkldnn.html) and
[CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html) policies per backend;
the recorder uses that API where available and the legacy API on older versions.
The runner tears down backends
and releases model references before the next cell. Artifact contents are checked
before loading, after loading and after inference. Source changes during the run
invalidate its verdict.

Concurrency is a bounded closed-loop probe over one serialized model runner. A
request's latency includes its wait for that runner and its whole completed call;
it is not divided by batch size. Threadpool admission waiting is excluded. This
does not claim to measure HTTP authentication, parsing, admission or network SLOs.
Use the [HTTP benchmark](microbatch.md) for the serving path. Model loading is
outside request samples, while first-call operator initialization may be cold.
Small samples cannot certify production p99, even when a numerical gate passes.

A saved baseline must cover the same corpus, completed cells, execution profiles,
artifacts, runtime precision and calibration policy. Its observations are checked
against their raw answers. Reviewed gate limits cannot silently be loosened in a
candidate. Source versions may change, because detecting their output drift is
the purpose of the comparison. Store the reviewed JSON in version control or an
artifact store with its digest. A passing report qualifies only its stated cells,
corpus and explicit criteria; it does not certify every deployment or checkpoint.

For programmatic diagnostics, `qualify()` accepts caller-owned runners with
`predict` and positional `predict_batch` methods only after explicit
`allow_unverified_identity=True`. Such runners cannot claim production
qualification. `run_matrix()` constructs verified local adapters.

## Recorded local qualification

The [5 November CPU artifact](https://github.com/tgundhus/Veya/blob/main/research/results/qualification_cpu_windows_20261105.json)
uses the existing English checkpoint and its existing ONNX export on Windows,
with four threads, two repetitions, batch sizes 1/4 and concurrency 1/4. Both
backends completed all 16 profiles over 18 evaluation cases. Source and artifact
identities remained unchanged. There were no label/operational-flag flips and no
observed drift in returned probabilities, scores, confidence or action probabilities. One case had
a measured top-two gap of 0.0442.

**The strict quality gate failed.** Choice agreement with the five explicit
fixture labels was 0.60 against a minimum of 0.80 in every profile. Noul agreement
was 0.80, score-within-0.25 was 0.80, and score MAE was 0.08264. Both cells are
therefore marked failed and `production_qualified` is false. These results expose
a checkpoint weakness on the fixture; thresholds and gold labels were not changed
to manufacture a passing release. This small synthetic corpus is not a business
quality certificate or a production latency SLO.

```bash
python research/scripts/qualify_local.py --checkpoint /models/veya \
  --onnx /models/english.onnx --out qualification.json
```

The harness records raw observations, per-call latency and forward evidence,
checkpoint/tokenizer/graph digests, numerical policy and source identities. Its
nonzero exit for a failed gate is expected and should stop a release pipeline.

The [follow-up investigation](reports/checkpoint-qualification-repair-2026-11-05.md)
reproduces the errors and evaluates an optional correction on the same fixture,
with unchanged labels and gates. Use `--decision-policy context-balanced-v1`
to select the corrected path in the local harness.
