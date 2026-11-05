---
title: Veya decision engines review, 5 November 2026
description: Source-based comparison of Laya, Kev, Laya-MLX, NanoJev and Nimble, with Veya speed measurements and durable decision replay controls.
---

# Veya: speed and stable decisions

Veya is the new project and distribution name. The repository is
[tgundhus/Veya](https://github.com/tgundhus/Veya), with Python distributions `veya`,
`veya-client` and `veya-onnx`. Existing `laya`, `laya_client`, `laya_onnx`, command
names and native SDK namespaces remain compatible. Upstream checkpoint identifiers
and attribution remain Laya. Historical benchmark JSON, logs and recorded paths
retain their original identities so the evidence is not silently rewritten.

The two goals are separate: reduce the cost of inference/cache lookup, and retain
the first final decision for exact repeats. Prefix/token/KV caches help the first
goal; they do not establish the second. A deterministic argmax can still change
when precision, batching, calibration or artifacts change a borderline probability.

## Repositories verified

These are the five requested projects, not a claim that they are the global top
five. Metadata was checked on 5 November 2026. A repository push timestamp can
reflect another branch or metadata activity; the table uses default-branch commits.

| Repository | Stars observed | Reviewed main HEAD | Main commit date | Assessment |
|---|---:|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | approximately 35k | `5a6dc4b809659dab3b9b3062f5e3902db983818a` | 5 November | Closest upstream; release 0.4.24 |
| [Kev](https://github.com/jaredpalmer/kev) | 9,839 | `2074cc44256200bb37540103fa4572901faa10f3` | 22 October | Causal Qwen decision models and serving optimizations |
| [Laya-MLX](https://github.com/mizorewww/laya-mlx) | 7,069 | `2159c6c9988e4cbfd6281db1888cef45de09cea9` | 29 October | Native Apple implementation of Laya architecture |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | 2,579 | `45b629283ddeb3b3784806f352bbf5e97f71dd40` | 16 October | Decision-head and invariance research |
| [Nimble](https://github.com/bespokelabsai/nimble) | 2,286 | `4edb0703a2e75d0170c3a45645a1961f4d1a2867` | 14 October | Causal structured scoring and contrastive evaluation |

Kev was pushed on 3 November, but its reviewed main HEAD is an October image
optimization; do not describe that as a November model-code commit. NanoJev and
Nimble's latest main commits update documentation. Their predictor/scorer work is
older. Stars and timestamps do not establish quality, speed or active code development.

## Laya changes after our last comparison

The [30 October review](upstream-review-2026-10-30.md) covered upstream
`118d361a774c9a28cd02a78d6cf4d2af0614f422` (0.4.18). The
[pinned comparison](https://github.com/NandhaKishorM/laya/compare/118d361a774c9a28cd02a78d6cf4d2af0614f422...5a6dc4b809659dab3b9b3062f5e3902db983818a)
contains 79 later commits. This review selects one small speed improvement;
it does not claim complete 0.4.24 parity.

**Adopted:** lazy text traversal and direct string/UTF-8 bytes handling, inspired
by [9580d43](https://github.com/NandhaKishorM/laya/commit/9580d43fafe16645ee1a03063d0c479c8f8532d1).
The previous helper materialized every text leaf before taking the routing budget.
It now stops once that budget is filled, while preserving leaf order, depth limits,
diagnostic paths and output bytes. Existing question-token caches are unchanged.

| `state_text` workload | Before median | After median | Ratio |
|---|---:|---:|---:|
| Short string | 0.211 microseconds | 0.068 microseconds | 3.09x |
| Short UTF-8 bytes | 0.337 microseconds | 0.177 microseconds | 1.91x |
| Nested list, 10,000 text leaves | 3,960.291 microseconds | 161.504 microseconds | 24.52x |
| Long first leaf followed by 10,000 leaves | 3,969.895 microseconds | 1.167 microseconds | 3,401.79x |

These are preprocessing measurements against a frozen eager reference, with nine
alternating samples on this Windows workstation. They are not model-inference
speedups or proof that Veya is the fastest engine. Fixed-seed tests include budget
edges, nested mappings, depth limits, cyclic structures, invalid UTF-8 and diagnostic
attribution. [Harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_state_text.py),
[raw samples and source hashes](https://github.com/tgundhus/Veya/blob/main/research/results/state_text_windows_20261105.json).

**Next upstream correctness candidates:**

- [46509b9](https://github.com/NandhaKishorM/laya/commit/46509b91b4ef832acaf264dc31d8b431b858fd53): explicit CUDA fp32/float32 handling and validation. Our CUDA helper currently ignores these values and typos; a caller can unintentionally retain bf16.
- [3401424](https://github.com/NandhaKishorM/laya/commit/3401424e7f755b0c94bd68fc537fddbb5820afdd) and [b87976e](https://github.com/NandhaKishorM/laya/commit/b87976eab51c05c15b604dd18cce8af9289163b5): TypeScript Unicode code-point lengths/cuts, relevant to emoji routing and Python/TypeScript agreement.
- [636abd0](https://github.com/NandhaKishorM/laya/commit/636abd0c5f53ccb10d107f6fa3883987efe728b3) and [5cc6d8c](https://github.com/NandhaKishorM/laya/commit/5cc6d8c8c03219e2a79e2f298855691f555ad8fa): wide-integer overflow handling for temperature and calibration input. Local calls with `10**400` still raise `OverflowError`.
- [2ea505c](https://github.com/NandhaKishorM/laya/commit/2ea505c4cc4ebea8d42c16baa062f55de2851c2c): reject nonfinite score fallbacks in guardrails. The current helper can return NaN for a NaN score.

Upstream also adds a [Vega/vegaml adapter](https://github.com/NandhaKishorM/laya/blob/5a6dc4b809659dab3b9b3062f5e3902db983818a/laya/vega.py),
held-out calibration and review routing. These are quality/backend research
directions with larger latency/memory costs, not a speed-first replacement for
the existing encoder path. Keep a future decoder backend optional. HTTP route-only
diagnostics, truncation warnings and native TypeScript long-input support are useful
separate capabilities. Existing Veya finite-weight confidence normalization and
per-request TypeScript batch budgets already cover some other upstream changes.

## Kev

Kev uses causal Qwen backbones, a shared state-before-question prefix and a pointer
readout. Its [server](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/serve.py)
has one inference thread, an asynchronous queue and queue-drain batching up to 64
requests. Veya already has a one-worker inference executor and heterogeneous
`Router.predict_batch`; bounded batching across separate HTTP requests is a useful
next experiment. Measure queue-inclusive p50/p95/p99 and cache-hit traffic, not
only saturated throughput.

Kev's [CUDA graph manager](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/cuda_graphs.py)
bounds resident graphs, defers capture until a shape is hot/idle and retains eager
fallback. Veya already buckets graphs, but captures unseen shapes synchronously
and retains an unbounded graph dictionary. Bounds and deferred capture are the
incremental ideas to evaluate on CUDA hardware.

Its token-budgeted prefix cache and [shared-prefix model](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/shared_prefix.py)
cannot be copied into question-conditioned bidirectional ModernBERT inference
without changing computation. That requires another backend or retrained/distilled
weights. Its cache is not final-decision retention. Group-disjoint out-of-fold
[calibration](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/calibrate.py)
and clustered uncertainty estimates would strengthen our evaluation pipeline.
The repository is Apache-2.0; model-specific terms still need checking before use.

## Laya-MLX

The [native model](https://github.com/mizorewww/laya-mlx/blob/2159c6c9988e4cbfd6281db1888cef45de09cea9/laya_mlx/model.py)
uses MLX arrays and fast scaled-dot-product attention. Its agent supports optional
whole-model compilation, shape padding, bounded batches and explicit completion
with `mx.eval`. This is the most direct optional Apple backend candidate, preserving
Veya's routing and final-decision store above it. It needs Apple Silicon/macOS and
Python 3.11+, so it cannot replace the portable Python 3.10/Windows/Linux path.

Its prepared-question cache duplicates functionality Veya already has. Its
[qualification](https://github.com/mizorewww/laya-mlx/blob/2159c6c9988e4cbfd6281db1888cef45de09cea9/docs/QUALIFICATION_0_4_12.md)
reports same-precision FP32 p50 improvements of 1.06–1.93x on an M3 Max against
its pinned upstream baseline. Those are publisher results, not measurements on
Veya. Headline FP16 latency combines hardware/precision choices. No Apple machine
was available here. The repository is Apache-2.0. Core ML discussed in its docs is
a sibling project, not part of the reviewed MLX implementation.

## NanoJev

The [production predictor](https://github.com/TianyuCodings/NanoJev/blob/45b629283ddeb3b3784806f352bbf5e97f71dd40/scripts/predict_toy_decisions.py)
flattens complete candidate paths into batches and explicitly reports
`prefix_sharing=False`. Its Qwen3-0.6B decision/set-attention heads require different
weights and architecture; the tree-attention implementation is a research parity
experiment rather than the production inference path.

Adopt its invariance-test discipline: single versus batch, candidate permutation
mapped back, unrelated-state mutation, question/state order and precision, including
deliberately broken masks/positions as negative controls. Its
[trained invariance artifact](https://github.com/TianyuCodings/NanoJev/blob/45b629283ddeb3b3784806f352bbf5e97f71dd40/research/trained_invariance_timing.json)
records probability drift up to 0.01508 in BF16 versus approximately 0.00000446
in an FP32 diagnostic. No label flips occurred in that small publisher fixture;
this still illustrates why argmax alone cannot promise sticky borderline outcomes.
The repository license is MIT. Veya already has batches, pinned artifacts and
parallel-option layouts, so those are not new additions to copy.

## Nimble

The [MLX scorer](https://github.com/bespokelabsai/nimble/blob/4edb0703a2e75d0170c3a45645a1961f4d1a2867/nimble/scoring/parallel_scorer.py)
prefills a common causal prefix once and forks KV/hybrid array caches per field.
Its CUDA scorer repeats independent full prompts. If Veya adds a causal backend,
compute common prefixes after tokenization (BPE can cross string boundaries), fork
all hybrid state without mutating the prefill object and verify cached serial and
parallel outputs against independent scoring. Existing Veya scalar heads already
avoid full-vocabulary projection; prefix sharing needs different architecture.

The strongest evaluation inspiration is contrast families: base example, removed
evidence, counterfactual, paraphrase and distractor. Keep entire families together
and prevent train/held-out leakage. See its
[contrast construction](https://github.com/bespokelabsai/nimble/blob/4edb0703a2e75d0170c3a45645a1961f4d1a2867/nimble/datasets/contrastive_data.py)
and [split logic](https://github.com/bespokelabsai/nimble/blob/4edb0703a2e75d0170c3a45645a1961f4d1a2867/nimble/training/split_evidence.py).
Its curation replay caches training-data requests, not final inference decisions.
No repository-wide LICENSE was found at the reviewed commit; an MIT license in a
plugin skill does not establish terms for the whole repository. Do not vendor its
code without resolving that; independently implement general techniques.

## Stable decisions and the 49/51 requirement

Veya already stores the first winning result atomically and returns that value to
losing writers. Its existing test races 0.51 against 0.49 and checks that both
callers receive 0.51. SQLite survives ordinary restarts; same-process in-flight
deduplication and duplicate rows avoid redundant computation. Device, precision and
backend are deliberately excluded from keys, while request state, ordered questions,
budgets, checkpoint routing and policy identity participate.

**Added:** HTTP and MCP deployment support for no capacity eviction and an explicit
policy identity, using the same shared Router/cache builder:

```text
LAYA_CACHE_PATH=/persistent/decisions.sqlite
LAYA_CACHE_TTL=none
LAYA_CACHE_MAXSIZE=none
LAYA_CACHE_FINGERPRINT=review-policy-v1
```

Compose forwards the policy identity. The Nix service accepts a string capacity
such as `"none"` and an optional `cacheFingerprint`; its service validates values.
Existing defaults are unchanged: deployment caching is opt-in and ordinarily uses
one-day retention and 10,000 entries. Python already supported `ttl=None`,
`maxsize=None` and an explicit `fingerprint` on `DecisionCache`.

The new regression stores 0.51, closes SQLite, advances a simulated clock ten
years and substitutes a model revision that would produce 0.49. The same policy
replays 0.51 without a forward pass. Changing to `policy-v2` explicitly permits
reevaluation and returns 0.49. Unbounded retention trades eviction for growing storage;
use stable route names and a deliberate policy-version process.

**This is not an unconditional always-same guarantee.** Cache I/O failures and
corruption currently fail open and can recompute. Lost databases/backups, clearing,
changed request identity, explicit policy changes and downstream hooks can change
the result. SQLite uses WAL with `synchronous=NORMAL`; an acknowledged decision is
not guaranteed to survive every power loss. Stronger promised consistency needs
an opt-in required-store/fail-closed policy (including hook failure/timeout paths),
appropriate durability and recovery rules. Retaining the first wrong answer does
not make it correct; abstention and review remain separate policy decisions.

## Recommended implementation order

1. **Required durable replay:** refuse an uncommitted replacement answer when the authoritative store is unavailable. Enforce it independently of optional telemetry-hook error suppression. Test concurrent processes, failure injection, restart and policy changes.
2. **Model-free cold replay:** persisted hits currently load the checkpoint before cache hooks run. A verified policy/metadata manifest could avoid that cost. Moving hooks earlier without preserving key/calibration identity is unsafe.
3. **HTTP microbatching:** compare a bounded request queue against the current executor on mixed schemas/models/lengths, cold misses, 70% repeats and concurrent identical bursts.
4. **Bounded/deferred CUDA graphs:** measure first-seen-shape tail latency, resident memory and fallback behavior on actual GPU hardware.
5. **Optional MLX backend:** qualify same-checkpoint, same-precision parity and cache-miss latency on Apple hardware.
6. **New shared-prefix architecture:** separately evaluate quality, memory, calibration and token-length scaling; do not present a changed model as a transparent optimization.

Every speed claim should include checkpoint/source identity, completed calls,
warm/cold conditions, p50/p95/p99, throughput, peak memory, forward-pass counts and
decision/probability parity. The work here improves preprocessing and makes durable
replay configurable; it does not establish an overall fastest-decision-model title.

## Validation completed

The shared CI/release gate passed all 106 registered suites, including a pytest
group of 963 passed, one checkpoint-dependent skip and 48 passing subtests. The
updated API contract passed 835 checks. Native TypeScript passed 1,214 tests;
the renamed HTTP client passed 33 tests including an external installed-package
consumer. Python wheels built with names `veya`, `veya-client` and `veya-onnx`,
and their metadata/import contracts were checked. The Python HTTP client wheel
and rebuilt source distribution each passed 66 tests with one Windows privilege
skip, followed by real local HTTP integration in extended and strict modes.

The strict documentation build and crawl/canonical checks passed for 55 sitemap
URLs. Ruff, byte compilation and whitespace checks passed. Tests needing local
loopback sockets or worker temporary directories ran outside the sandbox after
the sandbox blocked those resources. Nix changes were checked by repository
contract tests, not a Nix evaluation; Docker, new CUDA graph behavior and an Apple
backend were not benchmarked here. No competing-model head-to-head inference
benchmark or new general accuracy claim is made.
