---
title: Single-pass English checkpoint repair
description: Local decision-head training, sealed acceptance cases and measured Torch, ONNX and MCP behavior.
---

# Single-pass English checkpoint repair

The [checkpoint investigation](checkpoint-qualification-repair-2026-11-05.md)
showed that content-free prior correction could repair the original examples,
at the cost of extra inference rows and weaker performance on some controls.
This follow-up learns a new decision head and runs it through the canonical
prediction path.

The first scorer-only experiment reduced development loss while preserving the
same rubric errors. A full decision-head experiment corrected both original
choice errors at epoch 12, but its ordinal control score declined. Review also
caught an incorrect slot-to-level mapping in a subsequent ordinal training
run. The final recipe uses semantic levels for rotated score options.

## Split and training protocol

The training bank has 138 labelled states across 18 families. The 30 calibration
states come from three separate families. Training caches the frozen encoder's
hidden states, rotates each question's options and updates the existing typed
decision head. The objective gives harder training examples more weight, keeps
a small anchor to the base distribution and penalizes ordinal expectation error.
The encoder, tokenizer and action head weights come from the base checkpoint.
Changes to scores can still change the inputs to the action head.

Epoch selection uses the two earlier fixtures as development sets. It ranks
aggregate shortfall against their accuracy targets, then worst and aggregate
accuracy, then cross-entropy. Temperatures come from the separate calibration
families. The selected candidate uses the same architecture and token budgets
as the base checkpoint.

The fresh acceptance corpus was sealed in `d6497ce2` before any candidate
acceptance predictions. It contains 36 labelled cases and 18 unlabelled cases
with unresolved conditions or competing evidence. Review verified separate
families and exact states across training, calibration, development and
acceptance. Its file digest is
`2576a8a76a366eefb7f94e1aa66d16e9cd27a87d52c0f1912196c43880383d96`.
The existing gate measures the returned top-two margin. The report also
identifies which predeclared unlabelled cases actually reached that margin.

The strict candidate runs use two required CPU cells, two repetitions, batch
sizes one and four, and concurrency one and four. The unchanged gates require
80% accuracy for each rubric type, zero label flips, probability and score
drift of at most 0.001, and at least one returned top-two gap of at most 0.05.
Score answers must fall within 0.25 of the rubric value. The fresh base run and
the candidate run on the older controls use serial profiles on both backends
with the same thresholds.

## Qualification results

Epoch 16 had the smallest combined accuracy shortfall on the development
sets. The frozen artifact completed all 32 profiles on each strict corpus
and two serial profiles on the older controls. Each cell used the same source
and artifact identities throughout its run.

| Corpus | Policy | Choice | Noul | Score within ±0.25 | Score MAE | Measured boundaries |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fresh acceptance | Base | 10/12 | 8/12 | 8/12 | 0.41953 | 3 |
| Fresh acceptance | Repaired checkpoint | 11/12 | 10/12 | 8/12 | 0.42967 | 1 |
| Original rubric | Base | 3/5 | 4/5 | 4/5 | 0.08264 | 1 |
| Original rubric | Repaired checkpoint | 4/5 | 5/5 | 4/5 | 0.11266 | 0 |
| Older development controls | Base | 9/15 | 13/15 | 8/15 | 0.40412 | 5 |
| Older development controls | Repaired checkpoint | 14/15 | 14/15 | 9/15 | 0.40559 | 4 |

**The repaired checkpoint remains experimental.** All 32 fresh profiles
missed the 80% score gate. The same four labelled cases fell outside tolerance
for both checkpoints, all with expected level zero. The repaired scores were
1.9848 and 0.9912 on the two observatory cases, 0.9425 on the greenhouse case,
and 1.0005 on the shuttle case. The older controls also missed the score gate.

The fresh run had one actual near-tie. `harbor-service-3` describes a caller
who supplies a vessel number but has yet to say what service they need. It
returned reception at 0.5059 and moorings at 0.4752, a gap of 0.0307, with
the same probabilities and label in every profile. This case has a routing
rubric. The closest of the 18 predeclared unlabelled cases had a gap of 0.1051.
Every profile had zero label and operational flag flips. The maximum fresh
probability and confidence drift was 0.0001, with zero score and action drift.

The original fixture passed all three accuracy gates and failed the boundary
gate. The nearest returned gap was 0.1258. The repair now routes the
missing-request example to other and assigns 0.2053 to authorization without
evidence. The explicit billing denial still routes to billing, and priority
without described impact still falls outside tolerance at 0.5629.

## Paired CPU latency

The benchmark warmed nine labelled prompts, then timed each one three times
per model with alternating order and four CPU threads. Each backend made 27
measured forwards per checkpoint, one per request. The weight comparison checked
every tensor's shape, dtype and value. Changes were confined to the existing
decision head, scorer and type embeddings.

| Backend | Base p50 | Repair p50 | Base p95 | Repair p95 | Median paired ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager CPU | 167.0 ms | 163.8 ms | 287.4 ms | 253.9 ms | 0.988× |
| Native ONNX CPU | 166.2 ms | 176.0 ms | 220.2 ms | 235.5 ms | 1.059× |

The repaired checkpoint avoids the previous policy's repeated inference rows.
The paired sample still shows a 5.9% higher median ratio in ONNX and is too
small to certify a latency SLO. GPU, other hosts, and broader workload mixes
need their own measurements.

## Deployment checks

Classic MCP and Code Mode accept a named local checkpoint with its paired ONNX
graph. The smoke test pins the graph, external tensors, checkpoint config and
tokenizer, prepares replay on a fresh router, and uses a distinct cache policy
identity. It observes cold retained answers, a new committed decision and a
subsequent cache hit. Both modes returned the two retained decisions with
zero models loaded, loaded only `english-repair` on the fresh miss, and recorded
one hit for the repeat. Classic advertised ten tools and Code Mode advertised
only `veya_execute`. The isolated ONNX wheel and rebuilt source distribution
passed the HTTP, MCP, sandboxed execution and verified replay consumer checks.

Use `LAYA_EXTRA_MODELS` and `LAYA_ONNX_PATHS` to register the matching directory
and graph. Set `LAYA_DECISION_POLICY=canonical` and send the registered name as
`model`. `LAYA_DEFAULT_MODEL` only controls the fallback for ambiguous routing.
The [MCP settings](../cli-mcp.md#startup-and-environment) describe artifact pins
and cache identity.

The [production capability report](production-capabilities-2026-11-05.md)
covers durable store identity, approved replay, HTTP hit scheduling, shared
routing, qualification and MCP Code Mode. The repair exercises the existing
local checkpoint registration, approval and decision interfaces.

## Review and evidence

Review-agent checked the training split, inference layout, ONNX artifact
identity, MCP changes and final evidence. The shared runner passed all 122
suites, including 1,630 pytest tests and 4,238 subtests, with one optional
TileLang skip. The installed wheel and sdist rebuild matched 36 packaged
runtime source digests. Ruff, compilation, and the strict documentation build
and site checks also passed.

The raw files under `research/results/` share the prefix `checkpoint_repair_`
and suffix `_windows_20261106.json`. They contain the `training`,
`acceptance_base`, `acceptance`, `original`, `development_controls` and
`verification` records, including source hashes, artifact hashes and samples.

## Reproduce

```bash
python research/scripts/train_checkpoint_repair.py --base /models/english \
  --epochs 16 --threads 8 --output models/english-repair-20261106
python -X utf8 scripts/export_onnx.py --model models/english-repair-20261106 \
  --output models/english-repair-20261106/english-repair.onnx
python research/scripts/qualify_local.py --checkpoint models/english-repair-20261106 \
  --onnx models/english-repair-20261106/english-repair.onnx --decision-policy canonical \
  --corpus research/evals/checkpoint-repair-acceptance-v1.json --out acceptance.json
python research/scripts/verify_checkpoint_repair.py --base /models/english \
  --base-onnx /models/english.onnx --candidate models/english-repair-20261106 \
  --onnx models/english-repair-20261106/english-repair.onnx \
  --corpus research/evals/checkpoint-repair-acceptance-v1.json --out verification.json
```

## Remaining gaps

- Remote MCP still issues individual HTTP requests for batch items. Grouping
  compatible items could save network round trips. That changes forward shapes
  and needs decision parity and older-server compatibility evidence.
- The earlier capability measurements show worker startup on warm Code Mode
  workflows and higher warm verified-cache lookup cost. They also show higher
  HTTP miss latency with the optional replay worker. Measure hit-heavy and
  miss-heavy traffic separately when comparing deployment configurations.
- Qualification now hashes every ONNX external tensor. For verified replay
  preparation, use a newly loaded session with pinned files in an immutable
  directory. A general replay approval can still capture external file stamps
  after an existing session was loaded. Load-time binding needs separate work.
- The strict matrix records Torch forward witnesses. ONNX forward counts in
  this report come from the separate paired benchmark's `session.run` counter.
  The full ONNX profile matrix still needs equivalent forward instrumentation.
- Score repair needs more training coverage for level-zero examples, especially
  explicit denials and routine events that mention severe outcomes. Calibration
  used ten English labels per type, with three options for choice and score
  and two for noul. Other widths and languages need independent calibration
  and acceptance evidence before using this checkpoint's confidence values.
