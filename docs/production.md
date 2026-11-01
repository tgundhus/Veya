---
title: Deploying Laya-Pro
description: Configure local HTTP and MCP inference, authentication, resource bounds, precision and cache retention for Laya-Pro.
---

# Deploying Laya-Pro

Pin a reviewed source commit and checkpoint revision, pre-download model files, and validate
your questions against labelled examples before serving traffic. A confidence threshold
should come from the errors and coverage measured on that workload.

## HTTP and MCP

`laya-serve` exposes single and batch inference plus `/health`. Bind to loopback for a local
application with `LAYA_HOST=127.0.0.1`. For a shared deployment, set `LAYA_API_KEY` and use
TLS at your reverse proxy. Public health probes report liveness; authenticated health reports
resident checkpoints and runtime details.

Set `LAYA_PRELOAD=0` for lazy loading, or preload only the required checkpoints with
`LAYA_MODELS=english,multilingual`. `LAYA_MAX_LOADED` bounds residency; `LAYA_THREADS` controls
CPU threading. Bound concurrency and work using `LAYA_MAX_CONCURRENT`,
`LAYA_MAX_TOKEN_BUDGET` and `LAYA_MAX_BATCH_TOKENS`. A `503` means the admission limit is
full; choose retry and backoff in the calling application.

MCP over stdio uses the same local models and cache settings. It can instead call your HTTP
server using `LAYA_BASE_URL`. In that mode, configure retention on the HTTP server.
See [HTTP configuration](http-api.md), [MCP](cli-mcp.md) and [Docker](docker.md).

For a smaller CPU installation, the optional [ONNX services](onnx-runtime.md#http-and-local-mcp)
use exported graphs with local checkpoint configurations and tokenizers. Install
`"./runtime/onnx[serve,mcp]"` from the checkout; the full engine keeps its current default.

## Cache configuration

Caching is opt-in. Both the HTTP and local MCP builders read:

| Variable | Behavior | Default |
|---|---|---|
| `LAYA_CACHE` | Enable an in-memory cache | disabled |
| `LAYA_CACHE_PATH` | SQLite file; a nonempty path enables caching | unset |
| `LAYA_CACHE_TTL` | Retention in seconds; `none` keeps entries indefinitely | `86400` |
| `LAYA_CACHE_MAXSIZE` | Target bound for stored decisions | `10000` |
| `LAYA_CACHE_RENEW_ON_HIT` | Renew retention on use | disabled |

Fixed retention checks age from the stored decision. Renewal extends a frequently replayed
decision's lifetime and is unsuitable when every decision must be recomputed within a fixed
interval. A callable TTL policy is available through the Python `DecisionCache` API.

Use an absolute SQLite path and a persistent local volume. Separate processes can share
that file; separate machines need a store designed for that purpose. SQLite WAL requires
filesystem locking, so avoid sharing the file over a network filesystem. Process-local
request coalescing does not prevent every process from computing a simultaneous miss;
the shared store resolves which decision wins.

Inspect `cache_info()` or authenticated runtime status for hits, misses, conflicts and errors.
If the store fails, the engine computes the request again, which may produce a different answer.
The memory bound is enforced on each write. SQLite checks it periodically to amortize database
maintenance and can temporarily exceed it; call `prune()` for immediate enforcement.

## Retention and deletion

The built-in cache stores hashes and result payloads, including labels, probabilities and
metadata. It omits original request text. User-supplied labels or hook-added result fields can
still contain sensitive data, so the database belongs within your application's data policy.

TTL prevents reuse after expiry. `cache.prune()` deletes expired rows; `cache.cache_clear()` deletes
all stored entries. Neither guarantees erasure from SQLite journals, free pages, snapshots or
backups. Schedule pruning for an idle process through your host's existing maintenance system,
and manage database checkpoints, file disposal and backup retention separately if erasure is
required. Close the cache before removing its files.

The SQLite row limit does not bound the physical file size. Deleted rows can leave reusable
free pages, and WAL files can grow until checkpointed.

## Precision and validation

fp32, fp16, bf16 and int8 can produce different probabilities and decisions. Keep a reference
dataset and compare accuracy, decision agreement, latency and memory before changing precision,
backend, batch shape or token budgets. A shared cache can replay the same answer across precision
modes, but it can also retain an incorrect answer computed in one of them.

Revision, subfolder, calibration and inference configuration contribute to automatic cache
identity. Change a custom `fingerprint` when your policy changes; reusing it deliberately can
retain old decisions across model upgrades. Keep source and model versions in deployment records.

Use the [evaluation harness](evals.md) for quality gates and the [benchmark report](reports/production-review.md)
for the checks performed on this branch. Synthetic models test execution and interfaces; they
do not certify pretrained quality or production latency.
