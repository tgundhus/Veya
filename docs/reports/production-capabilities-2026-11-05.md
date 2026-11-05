---
title: Production capability implementation, 5 November 2026
description: Verified cold replay, HTTP cache scheduling, shared routing, release qualification and local MCP Code Mode, with measurements and independent review.
---

# Five production capabilities

This implements the next steps from the [runtime adoption review](runtime-adoption-2026-11-05.md).
Existing deployments retain their defaults. Replay approvals and MCP Code Mode are opt-in;
the HTTP cache fast path operates only with approved metadata. Execution remains local.

## Implemented interfaces

| Capability | Interface | Contract |
|---|---|---|
| Verified durable replay | `Router.export_replay_manifest`, `validate_replay_manifest`, `try_replay`; `laya-replay` | Exact approved hits require neither model construction nor inference-library imports. Misses verify the loaded model against approval before inference. |
| Store identity | `DecisionCache(require_existing=True, expected_store_id=...)` | Refuse a missing, incompatible or replacement SQLite store before creating or modifying a replacement. Preserve its UUID separately. |
| HTTP cache scheduling | `LAYA_HTTP_REPLAY_FAST_PATH` | Verified hits bypass model collection and inference queues; misses retain bounded scheduling. Disable the extra worker for cache-poor traffic. |
| Shared routing | `veya-routing-v1-unicode15` | Pinned Unicode, canonical numeric state-key ordering and matching diagnostics for supported JSON inputs and routing controls. |
| Release qualification | `laya-evals qualify`, `qualify-compare`, `qualify-validate` | Local execution matrices and reviewed baselines, including accuracy, decision/probability/confidence/action drift, boundary coverage and latency gates. |
| Local MCP Code Mode | `laya-mcp-server --mode code`, `CodeModeRunner` | One MCP tool composes validated operations in bounded local JavaScript. Classic mode remains the default. |

[Replay setup and recovery](../replay.md), [HTTP scheduling](../cache-fast-path.md),
[routing and migration](../routing-parity.md), [qualification](../release-qualification.md),
[MCP Code Mode](../mcp-code-mode.md).

## Approval and execution boundaries

Replay metadata binds the store, policy identity, routing contract, calibration, token
budgets and source identity. ONNX preparation uses the official parser to enumerate external
weights, including nested graphs, sparse tensors and functions. The optional `replay` extra
is needed for ONNX preparation; consuming prepared approvals does not import ONNX. Local
identity follows the documented metadata/digest contract. The checksum detects corruption;
protecting trusted deployment files supplies authorization.

Arbitrary prediction hooks and callable retention policies keep their ordinary path.
Required storage failures remain terminal. Explicit policy fingerprints retain compatible
decisions; automatic identities include the routing-contract version. Previously ambiguous
keys for explicit model calls with per-language calibration are separated by their actual
calibration language. Old ambiguous rows cannot safely authorize those new keys.

MCP Code Mode keeps models and caches in the host. A fresh QuickJS worker receives JSON and
a capability API, with no Python callbacks, filesystem, network or environment bindings.
The parent independently enforces CPU/wall deadlines, admission and byte/call limits;
host validation bounds inference work. Timed-out model work retains admission until it
finishes. Required storage and transport failures abort outside guest exception handling.
This is a JavaScript capability boundary, not an OS sandbox for native code.

## Measurements and qualification

Benchmarks retain source/artifact identities and raw observations. Cold replay uses fresh
processes separately from warm hits. HTTP cases include all-hit, mixed and all-miss traffic.
Code Mode includes worker startup and compares classic individual calls and the existing
batch API. No LLM or network timing is assumed.

The existing English CPU fp32 checkpoint produced the same retained answers with
zero forward passes in both replay modes. Five alternating fresh-process pairs
measured these local medians:

| Measure | Ordinary cache | Approved model-free replay |
|---|---:|---:|
| Process startup, first retained answer and cleanup | 6,765 ms | 227 ms |
| First retained-answer operation after required imports | 4,481 ms | 8.82 ms |
| Process high-water memory | 1,939.8 MiB | 37.4 MiB |
| Warm direct cached prediction p50 | 59.5 microseconds | 375.8 microseconds |

Approval/source checks make warm calls more expensive. The improvement is cold
startup and model residency, not a blanket warm-cache speedup. The small tail sample,
filesystem caches and uncontrolled workstation load limit generalization.
[Raw cold/warm samples](https://github.com/tgundhus/Veya/blob/main/research/results/verified_replay_windows_20261105.json).

The HTTP comparison used 16 requests, concurrency eight and three paired repeats,
with identical required-cache policies and four CPU threads. It includes request
handling and queueing, but uses an in-process ASGI transport rather than a network.
Client latency starts after the client concurrency semaphore, so client admission
waiting is excluded. With sixteen observations per run, p95 and p99 select the maximum.

| Workload | Queued throughput | Cache worker throughput | Queued overall p95 | Cache worker overall p95 |
|---|---:|---:|---:|---:|
| All retained hits | 1,288.06 requests/s | 1,279.00 requests/s | 6.10 ms | 6.40 ms |
| 75% retained hits | 37.77 requests/s | 43.14 requests/s | 237.58 ms | 363.24 ms |
| All misses | 15.37 requests/s | 13.71 requests/s | 537.47 ms | 606.70 ms |

Mixed-traffic retained-hit p95 fell from 237.58 to 7.42 ms; miss p95 rose from
236.47 to 363.24 ms. The fast path grouped four missed rows into two forwards of
two rows instead of four forwards of one row. All-miss traffic kept two forwards
of eight rows but paid additional lookup/scheduling overhead. Every paired answer
matched exactly, with no label flips, probability drift or cache errors. Disable
the fast path when its workload tradeoff is unsuitable.
[Raw CPU HTTP samples](https://github.com/tgundhus/Veya/blob/main/research/results/http_replay_cpu_windows_20261105.json)
and the separate
[controlled scheduling experiment](https://github.com/tgundhus/Veya/blob/main/research/results/http_replay_synthetic_windows_20261105.json).

Code Mode was compared with the existing classic batch API on eight records and
three rotating runs. The fresh JavaScript worker is included in every workflow.

| Workflow | Classic batch median | Code Mode median | Forward work in both modes |
|---|---:|---:|---|
| Unique decisions | 636.86 ms | 838.98 ms | One forward, eight rows |
| Two dependent stages | 1,553.02 ms | 1,677.30 ms | Two forwards, sixteen rows |
| Prewarmed decisions | 0.53 ms | 192.70 ms | Zero forwards |

All nine paired workflow outputs matched exactly. The two-stage workflow reduced
outer calls from two to one and returned 272 rather than 4,666 JSON bytes. The
catalog fell from ten tools/26,214 JSON bytes to one tool/4,332 bytes. These are
composition and payload benefits; the measured local workflow was slower. No LLM,
token, network-speed or memory benefit is inferred from those byte counts.
[Raw Code Mode samples](https://github.com/tgundhus/Veya/blob/main/research/results/code_mode_cpu_windows_20261105.json).

Routing preprocessing has an explicit correctness cost. Ordinary short Python analysis
stayed approximately unchanged (10.487 to 10.421 microseconds); a support example changed
from 274.021 to 281.086 microseconds. Canonical numeric ordering requires examining mapping
keys: analysis of a 10,000-numeric-key state increased from 4.308 to 19.635 ms. Values remain
lazy. These are preprocessing timings, not encoder speed measurements.
[Raw routing samples](https://github.com/tgundhus/Veya/blob/main/research/results/routing_contract_windows_20261105.json).

Qualification refuses to mark a failed matrix production-qualified. The hand-annotated
corpus is an illustrative release fixture, not a representative business dataset or a
general accuracy benchmark. Preserve its quality thresholds; use independently reviewed,
family-disjoint data for the deployment's acceptance policy.

The final eager/ONNX run completed all 32 configured serial, batch, reversed-order,
unrelated-neighbour and concurrency profiles with unchanged sources and artifacts.
Returned labels, operational flags, probabilities, scores, confidence and action values
matched across all profiles. One case had a measured top-two margin of 0.0442.

The strict matrix **failed qualification**: both cells scored 0.60 choice accuracy
against the configured 0.80 minimum. Noul accuracy was 0.80, score mean absolute error
was 0.08264 and score-within-0.25 was 0.80. The report retains all 32 quality failures
and `production_qualified: false`; neither labels nor gates were adjusted. Its p95/p99
ceilings are unset, so it does not certify a latency SLO.
[Raw qualification report](https://github.com/tgundhus/Veya/blob/main/research/results/qualification_cpu_windows_20261105.json).

## Review and verification

Each capability received an independent read-only review using the review-agent workflow.
Regressions cover external ONNX data, source/configuration changes, namespace races,
calibration, callback serialization, cold imports, shutdown races, Unicode versions,
qualification false passes, CPU escapes, protocol spoofing and abandoned work.

CI/release gates install the optional dependencies needed to exercise these paths. Live
Python/TypeScript checks use a pinned Node toolchain alongside the supported Python matrix.
Standalone ONNX verification installs outside the checkout, exercises replay, HTTP and both
MCP interfaces, then rebuilds and rechecks the source distribution.

Local verification used Windows 11 and Python 3.12.14. The native TypeScript suite
passed 1,339 tests in 45 files; an offline packed-package consumer also verified
the exported routing contract, diagnostics, Unicode module and TypeScript types.
The HTTP TypeScript client passed 33 tests. The installed HTTP Python wheel and
rebuilt source distribution each ran 66 tests with one platform-specific skip,
plus real HTTP service checks. The standalone ONNX wheel and outside-checkout
sdist rebuild passed their base, replay, HTTP, classic MCP and Code Mode consumers;
all 35 packaged Python module digests matched the final sources. `pip check`,
the required Ruff selection and compilation passed.

The shared CI/release runner passed all 117 suites: 85 script suites and 32 pytest
files. Its pytest group passed 1,491 tests and 4,238 subtests, with one TileLang
dependency skip. The public hook/API contract passed 902 checks. Final-suite
integration fixes updated the dependency-audit allowlist for the two new extras
and replaced warning capture that accidentally inspected lazy optional Transformers
modules; the replay assertion and callback-serialization negative controls remained effective.
The strict clean documentation build emitted no `griffe:` warnings, and the
documentation identity/crawl check passed all 63 sitemap URLs.

The Linux/Python 3.10–3.13 and additional Windows jobs are configured in CI; these
local results do not claim those remote jobs have already executed.

The implementation is split into coherent commits:

| Commit | Scope |
|---|---|
| `3020d257` | Shared Python/TypeScript routing contract, diagnostics and migration tests |
| `96097d32` | Local release qualification and reviewed baseline comparisons |
| `bad10f92` | Verified replay, guarded store identity and cache-aware HTTP scheduling |
| `ed5cd5ce` | Bounded local MCP Code Mode and standalone distribution coverage |

Measurements record source digests independently of commit labels. The benchmark
sources and all packaged runtime modules were checked against the frozen implementation;
the evidence commit adds reports and verification refinements without changing runtime code.
Independent review recomputed all 319 source digests across the six routing and
capability artifacts with no mismatches.

Publication and deployment remain separate. These CPU measurements do not qualify CUDA,
MLX or competing models. An overall fastest-engine claim still requires a comparable
workload, accuracy target and hardware baseline.
