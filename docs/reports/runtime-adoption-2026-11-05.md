---
title: Veya runtime adoption, 5 November 2026
description: Implemented decision-retention and serving improvements, independent review, measurements and remaining hardware qualification.
---

# Runtime adoption after the engine comparison

This records the implementation following the [five-repository comparison](decision-engines-2026-11-05.md).
The work prioritizes exact decision replay and measured local throughput. All features run
locally; neither required storage nor HTTP microbatching is enabled by default.

## Adopted changes

| Area | Result | Basis |
|---|---|---|
| Required decision storage | `DecisionCache(required=True)` returns an authoritative stored answer or raises `DecisionCacheError`; HTTP and MCP expose sanitized unavailable errors | Extends Veya's existing first-writer cache for policies where recomputation is unacceptable |
| HTTP microbatching | Optional, bounded collection of compatible singles; shared worker, cancellation isolation and conservative individual fallbacks | Serving pattern used by high-throughput local inference engines, including Kev |
| CUDA precision | Explicit fp32/float32 disables internal and ambient CUDA autocast and refuses half-only fast kernels | Selected upstream Laya corrections |
| Unicode routing | TypeScript budgets/counts use complete Unicode code points and stop traversal at the budget | Selected upstream Laya corrections, with bounded allocation |
| Numeric robustness | Wide integer temperatures/calibration and invalid guard scores follow existing validation/fallback policy | Selected upstream Laya corrections |

The [earlier text-traversal optimization](decision-engines-2026-11-05.md#laya-changes-after-our-last-comparison)
is already implemented. Its large synthetic preprocessing ratios are not inference-throughput claims.

## Retaining a borderline decision

For a policy that must keep its first stored 0.51 answer even if later inference would
produce 0.49, use persistent SQLite storage, `required=True`, `ttl=None`, `maxsize=None`
and a deliberately versioned fingerprint. The equivalent deployment settings are:

```text
LAYA_CACHE_PATH=/persistent/decisions.sqlite
LAYA_CACHE_REQUIRED=1
LAYA_CACHE_TTL=none
LAYA_CACHE_MAXSIZE=none
LAYA_CACHE_FINGERPRINT=review-policy-v1
```

Required mode uses SQLite FULL synchronization, validates authoritative commit read-back,
and refuses successful predictions after storage errors or prediction-hook deadlines.
Ordinary transformations run before commit; later mutable model objects cannot alter
the returned canonical answer. Cache failures cannot be suppressed with `hooks_raise=False`.
Install required storage on the Router when using a Router.

The guarantee covers the retained decision for the same keyed request and policy. It
does not establish accuracy, preserve erased records, make memory storage persistent,
or prevent reevaluation after expiry, eviction or a policy change. Current routing,
usage and confidence/abstention presentation can differ between calls. Back up and
preserve the authoritative database. [Full contract](../consistency.md).

## Measurements

The cache benchmark compares optional and required storage using a controlled predictor.
It measures Router/cache orchestration, including SQLite synchronization, rather than
neural inference. Warm hits replay 0.51 after the predictor changes to 0.49, with no
predictor calls. Unique commits exercise writes. The harness alternates policy order
and retains raw timings, source hashes and percentile methodology.

| Operation | Optional p50 | Required p50 |
|---|---:|---:|
| Memory exact hit | 29.5 microseconds | 43.2 microseconds |
| Memory unique commit | 43.9 microseconds | 66.2 microseconds |
| SQLite exact hit | 34.7 microseconds | 49.1 microseconds |
| SQLite unique commit | 85.2 microseconds | 595.8 microseconds |

Each case includes 3,500 hits and 700 unique commits over seven alternating
repetitions. Required SQLite writes cost more because they use FULL synchronization
and authoritative read-back. All replay checks passed; errors were zero.

[Cache harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_required_cache.py),
[cache samples](https://github.com/tgundhus/Veya/blob/main/research/results/required_cache_windows_20261105.json).

The microbatch benchmark measures queue-inclusive ASGI latency and throughput, with
both a synthetic fixed-overhead fixture and an existing English CPU fp32 checkpoint.
It includes unique requests, prewarmed repeats and identical fresh bursts. A duplicate
cache-backed group executes individually to preserve the cache's single-miss behavior.
Results are workload-specific; batching can change probabilities and add latency to
cache hits. Keep it disabled until representative traffic benefits.

[Microbatch behavior and reproduction](../microbatch.md),
[CPU checkpoint samples](https://github.com/tgundhus/Veya/blob/main/research/results/microbatch_cpu_windows_20261105.json),
[synthetic samples](https://github.com/tgundhus/Veya/blob/main/research/results/microbatch_synthetic_windows_20261105.json).

## Review and remaining qualification

Independent read-only reviews use the review-agent workflow. Regression checks cover
late hook mutation, unsuppressible storage failure, nested cache placement, scan provenance,
ambient autocast, bounded queues, cancellation, token budgets and custom-agent fallbacks.
The shared CI runner includes the new required-cache and microbatch suites.

Validation passed across the shared 85 script suites and 24 pytest files
(1,007 pytest cases passed, one skipped), including rerunning the corrected Compose
documentation check. Final checks include 23 required-cache tests, 839 hook API
assertions and 1,234 native TypeScript tests. Ruff, compilation, strict clean
documentation build and site checks pass. The rebuilt standalone ONNX wheel
replays required decisions and recognizes native batching without importing
Torch, Transformers or the full `laya` package.

The implementation is split into these independent commits:

| Commit | Change |
|---|---|
| `bdba12bc` | Wide integer temperatures and calibration |
| `0a91293e` | Finite guard-score fallbacks |
| `4cc49ca1` | Explicit CUDA full precision and ambient autocast isolation |
| `fd97dd03` | TypeScript Unicode code-point routing |
| `fcc559cb` | Required authoritative cache, transports and storage benchmark |
| `24f05d6a` | Opt-in bounded HTTP microbatching and checkpoint benchmarks |

CUDA numerical accuracy/performance and native Apple MLX execution require their target
hardware. The Unicode changes repair demonstrated Python/TypeScript mismatches; existing
leaf/line-analysis differences still prevent a claim of complete cross-SDK parity.

Model-free cold replay needs a persistent policy/routing manifest that can identify a
decision before model construction. Shared causal prefix execution needs a different
model architecture and training. Bounded CUDA graph capture and a native MLX backend
need hardware qualification. These remain separate work; importing their implementations
would not make existing encoder checkpoints safely support those features.

There is no evidence here that Veya is the fastest decision engine overall. Compare
queue-inclusive latency, throughput, retained-decision agreement and accuracy on the
same real workload and hardware before making that claim.
