---
title: Laya upstream review, 25 November 2026
description: Review of Laya 0.5.4, teacher-assisted training, runtime performance, and SDK changes for Veya.
---

# Laya upstream review, 25 November 2026

The upstream heads and tags were fetched from [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya).
Main and tag `v0.5.4` both resolve to `60a6ce910b4a2edb2b77e00d8fddeb24349610c8`.
The [range since 0.4.18](https://github.com/NandhaKishorM/laya/compare/118d361a774c9a28cd02a78d6cf4d2af0614f422...60a6ce910b4a2edb2b77e00d8fddeb24349610c8)
contains 227 commits, including 152 non-merge commits. Of those, 148 commits (100 non-merge)
follow `5a6dc4b809659dab3b9b3062f5e3902db983818a`, the 0.4.24 snapshot in the
[5 November comparison](decision-engines-2026-11-05.md). This is a source review and selective
adoption against Veya's `origin/main` at `2dd6c79da2cc73dbc88785da50201bdf268f31c6`.

The fetched `learn-teacher` and `integration-0416` heads are already ancestors of upstream main,
as are the dependency update heads. The divergent `research` branch has one unique September
commit with T4 and CPU language sweep scripts and results.

| Area | Upstream changes | Fit for Veya |
|---|---|---|
| Classic Python runtime | Hookless router fast path, batch-size and legacy fallback fixes, Jev truncation warning, wide numeric inputs and CUDA precision control. Lazy state traversal was covered on 5 November. | Adopt batch count validation and skip unused hook context creation after hook resolution. Veya's language fallback and finite-weight handling remain in place. |
| CPU execution | Selective final-head outputs and exact local attention windows ([`511a041`](https://github.com/NandhaKishorM/laya/commit/511a041c61ff55640d97d4649c15855bec2ff830), [`41e2c3a`](https://github.com/NandhaKishorM/laya/commit/41e2c3a84725575167be9d44372f3be0b6bfe492)). | Benchmark separately on Veya checkpoints. Upstream's official long-input p50 falls from 10.787 s to 4.836 s for its longest row, but short CPU workloads vary. Its 200 case configurations preserve top choice, with up to 0.0001 probability movement. Those numbers require a Veya gate and real-checkpoint parity run before rollout. [Upstream CPU report](https://github.com/NandhaKishorM/laya/blob/60a6ce910b4a2edb2b77e00d8fddeb24349610c8/benchmarks/results/cpu/README.md) |
| Calibration and fitting | Vectorized calibration validation, temperature grouping, threshold selection and binning. Vega fits heads from CSV/JSONL, holds calibration rows apart and serves a learned head after comparison with the base engine. | Veya already has CSV/JSONL training, held-out calibration, and family-disjoint release qualification. Port measured savings selectively so Veya's report fields, split arithmetic, threshold ties and malformed-input behavior survive. There is no corresponding upstream change to classic `train.py`, `train_cli.py` or `evals.py` in this range. [Calibration optimization](https://github.com/NandhaKishorM/laya/commit/a1b73e7b1e2970277469acc33dc6b002b6097323) |
| Vega runtime and policy artifacts | `vegaml` became the default in 0.5.0. The reader adds multimodal inputs with a video sampler, review routing from fitted bands and prediction sets, signed ZIP policy import, stricter gold validation and label restoration. | Evaluate the reader separately with score, choice, abstention, calibration, CPU latency and memory witnesses. Its trained heads and signed Studio policy bundles belong to a different backend and artifact format from Veya checkpoints and local replay manifests. [Vega guide](https://github.com/NandhaKishorM/laya/blob/60a6ce910b4a2edb2b77e00d8fddeb24349610c8/docs/vegaml.md) |
| Teacher feedback | `Learner` refits a head on confirmed labels. `Escalator` asks a teacher for uncertain records and audits a sample of accepted records. A local callback or caller-hosted vLLM can supply labels. Upstream offers hosted proxy and Studio services separately. As of 0.5.4, `Learner`, `Escalator`, and `vllm_teacher` load from optional PolyForm Noncommercial `vegaml-learn`; Laya itself remains Apache-2.0. | Use Veya's local teacher evidence, evaluation and soft-target training formats. Teacher agreement measures imitation; release accuracy and calibration still need separate human gold. Preserve train/calibration/evaluation family separation, and promote a new checkpoint explicitly after qualification. [0.5.4 licensing change](https://github.com/NandhaKishorM/laya/commit/dc5aa9e6505b49c9ff0c19d9be153f8f10eb0e9c) |
| Native SDKs and exports | TypeScript long scans, Unicode code-point/decode parity, batch grouping/budgets/gates, hooks deadlines, diagnostics and revision inheritance. Java, .NET, Rust, Go and C++ gain revised Romance language evidence; .NET and Rust exporters verify against the actual export model. | Veya already has parts of TS diagnostics and per-request budgets. Prioritize artifact selection and behavior defects. Verify long scans against Veya's provenance and cache semantics, and language changes across all supported SDKs and short English controls. Go, Rust and C++ remain separate ports. |
| HTTP and integrations | Route-only diagnostics over HTTP, TypeScript HTTP batch and routing clients, optional native LangChain routing, per-bucket tool gating, numeric guard hardening and stability hook vote accounting. | Veya already has the TS HTTP batch client and local/MCP routing diagnostics. An HTTP route endpoint would need admission, authorization, replay and both server backends reviewed together. The per-bucket tool fix targets an upstream selector absent from Veya. Adopt applicable shared guard fixes. |
| Tooling and CI | Workflow pins, Gitleaks fingerprints, expanded audit for Vega, npm ONNX Runtime/Vitest updates, optional Rust ONNX Runtime download, ONNX opset autodetection and export fixture/cleanup fixes. | Keep Veya's scoped dependency audit, Agno floor, package names and workflow controls. Review dependency upgrades with their own lockfiles and CI. Check ONNX exporter fixes against Veya's production and native tools independently. |

Veya's current release process qualifies an explicit checkpoint on fixed corpora. A teacher
workflow should record model/revision and independent repeats, send only selected states through a
caller-controlled label source, report coverage and disagreement, and retain rejected labels for
review. A live-learning experiment would also need evaluation and rollback on a separate store,
without allowing requests to modify the checkpoint that normal inference uses.

## Veya teacher training

[`veya-train`](../finetune.md#learn-from-saved-teacher-answers) accepts saved JSONL teacher repeats for
choice, yes/no and score decisions. The frequency of each observed answer becomes a soft target
for the existing local trainer. Every repeat must answer the same questions in the original schema.
The calibration split keeps the same tokenized state together across sibling questions and duplicate
rows, and evaluation flags overlap on those states even when the questions differ. Dataset authors
still need to hold out families such as conversations, paraphrases and counterfactual pairs.

A Windows CPU smoke run using the local English repair checkpoint's tokenizer prepared 36 usable
decisions from 12 synthetic cases and three repeats per case. The 33 training items and three
calibration items were separated by whole case. A separate labelled case produced three usable
evaluation decisions. The training suite also fine-tunes a tiny local checkpoint, writes the
candidate, and verifies the held-out report. These checks establish that the pipeline works with all
three types. They measure neither frontier agreement nor improved task accuracy on real traffic.

For a real trial, keep one independent human-labelled acceptance set and use the same base checkpoint
and budget for two candidate runs. Compare a random teacher sample with one selected from local
abstentions or low confidence, retaining a random audit of accepted decisions. Report teacher calls,
coverage at the same error rate, latency, correctness, and calibration on the acceptance set. Pass the
chosen candidate through [release qualification](production-capabilities-2026-11-05.md) before changing
the deployed policy. The training command writes a separate candidate so replay and teacher
experiments can be evaluated before promotion.

## Calibration performance

Veya fits temperatures from validated records without repeating validation for each type and size
bucket. Abstention fitting evaluates confidence ties as a group and binning counts each record once.
On Windows with two CPU threads, 12,000 seeded float32 records and 2,400 held out for temperature
evaluation yielded these medians over seven alternating pairs.

| Workload | Before | After |
|---|---:|---:|
| Temperature fitting and held-out ECE | 1,184.7 ms | 499.1 ms |
| Probability binning | 259.9 ms | 134.2 ms |
| Raw abstention thresholds | 280.5 ms | 143.4 ms |
| Binned abstention thresholds | 276.2 ms | 147.5 ms |

The candidate was faster in all seven pairs for each workload. Temperatures, ECE, bin values,
thresholds and counts matched exactly. The 1,200-record workload also showed exact parity and
smaller savings. These timings measure calibration after logits are available.
[Benchmark harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_calibration_fit.py) ·
[Raw measurements](https://github.com/tgundhus/Veya/blob/main/research/results/calibration_fit_windows_20261125.json).

## Runtime and SDK changes adopted

- Torch, ONNX and Router batch and long-scan entry points reject invalid batch counts before inference.
  `None`, zero and negative integers retain their existing uncapped behavior.
- Native TypeScript inherits a shared checkpoint pin when a per-model revision is blank. Aliases of
  one checkpoint now group into the same routed batch. The regression uses one agent call after the
  fix where it previously used two. The native suite passed 1,341 tests and TypeScript type checking.
- Routing without active hooks avoids allocating a hook context after composition and timeout
  validation. A 64-request explicitly routed batch fell from 308 to 186 microseconds median on
  Windows CPU with nine alternating pairs and identical routing decisions. This measures routing
  overhead before model inference.
- LangChain's opt-in `remote_batch` sends one HTTP request for a list of states with shared questions.
  A regression reduced three requests to one with the same ordered outputs. An authenticated test
  against the Veya app verified one server batch dispatch with the same language and confidence
  controls. Requests made with `return_exceptions=True` still run individually.
- LangChain, CrewAI, Agno and LlamaIndex routing honor explicit core abstention. A configured
  fallback handles those cases. Otherwise the adapter raises `VeyaLowConfidenceError` before
  dispatching a target. An unevaluated gate can report missing confidence.

## Verified fixes

- Integration guards now report an out-of-range integer using the malformed decision response
  error. A mocked CrewAI HTTP response with `10**400` previously escaped as `OverflowError`.
  The CrewAI script passes all 249 checks. [Upstream guard change](https://github.com/NandhaKishorM/laya/commit/8956f1e704f3c44473dff5577d18c0e9e02a0135)
- The .NET fused ONNX export verifies against the selected PyTorch model already in memory. It
  no longer scans the cache and rebuilds an unrelated reference model. A local tiny-checkpoint
  export on Windows CPU puts a second compatible checkpoint beside the export. The prior shape
  sweep reported a maximum logit gap of `1.000` over 24 shapes. The fixed sweep stays within
  `0.001` for logits and `0.01` for action logits. This check uses a real ONNX export and runtime
  session with test weights. [Upstream exporter change](https://github.com/NandhaKishorM/laya/commit/1d9aaae8b352e58a41baf6b7cb50344c6fb26cf5)
