---
title: Verified replay without loading models
description: Prepare local policy metadata for exact durable cache hits without constructing inference models, and pin the identity of the authoritative SQLite store.
---

# Verified replay without loading models

A retained exact decision can be returned after a restart without importing an
inference runtime or loading model weights. This is opt-in: a trusted local manifest
records the approved routing, calibration, token budgets and checkpoint identity.
The ordinary cache and inference paths remain the default.

## Prepare an approved policy

Use absolute paths and preserve the same Router construction options in preparation
and deployment. Preparation loads the checkpoints to approve their actual metadata.
Only resident models enter the manifest; other registered models keep the ordinary
prediction path.

```python
from veya import DecisionCache, Router

models = {"english": "/models/english"}
with DecisionCache(
    "/persistent/decisions.sqlite", required=True, ttl=None, maxsize=None,
    fingerprint="support-policy-v1",
) as cache:
    with Router(models=models, hooks=[cache], performance_mode="default") as router:
        router.load("english")
        approved = router.export_replay_manifest("/persistent/support.replay.json")
        print(approved["store_id"])  # retain this separately in deployment configuration
```

The export is written through a temporary file, flushed and atomically replaced.
The manifest is finite, bounded JSON with a checksum and typed values; it is never
unpickled or executed. The checksum detects corruption, not an unauthorized writer.
Protect the manifest and checkpoint directories as deployment configuration. Prepare
while model files and calibration are stable.

Remote checkpoint sources require an explicit immutable 40-character revision.
Local sources bind their resolved directory and load-time artifact metadata. A
relative path that resolves elsewhere after a working-directory change is refused.

## Serve or predict after a restart

```python
with DecisionCache(
    "/persistent/decisions.sqlite", required=True, ttl=None, maxsize=None,
    fingerprint="support-policy-v1", expected_store_id="UUID_FROM_PREPARATION",
) as cache:
    with Router(
        models=models, hooks=[cache], performance_mode="default",
        replay_manifest="/persistent/support.replay.json",
    ) as router:
        assert router.validate_replay_manifest()
        result = router.predict(state, questions, model="english")
```

`validate_replay_manifest()` checks every approved record without reading decisions
or loading models. It returns `False` when replay is disabled or custom hooks make
the optimized path unsupported. Invalid configured store, policy, source or metadata
identity raises `DecisionCacheError`. Local HTTP and MCP builders run this validation
after installing the cache so invalid deployments fail at startup.

Both `predict` and `predict_batch` use approved exact hits before loading a model.
Mixed batches load only for their remaining misses. A newly loaded model must match
the approved metadata before a miss can be inferred or committed. Store read/decode
and renewal failures remain fatal in required mode. Optional storage preserves its
existing fail-open behavior; manifest integrity and policy mismatches are always fatal.

`Router.try_replay(...)` exposes the same verified lookup to integrations. It returns
a result on a hit and `None` on a miss or unsupported optimized path. It creates no
in-flight claims, waits for no miss owner, and does not increment miss counters.
Successful hits increment hit counters and honor expiry and renewal. Callable
retention policies retain their ordinary hook execution path, including error
suppression and serialization. Confidence/abstention presentation is reapplied for
the current call.

## Deployment preparation command

The public preparation command uses the same environment builder as the local HTTP
service. Configure the intended SQLite path, policy, checkpoint paths and backend
first, then run:

```bash
python -m veya.replay --output /persistent/support.replay.json --models english
```

The full distribution also installs `veya-replay`. The standalone ONNX distribution
installs `veya-onnx-replay` with the same arguments and its ONNX backend default.

Preparation temporarily disables preload and ignores an older replay manifest while
approving its replacement. Existing-store and expected-identity guards remain active.
Choose `VEYA_MAX_LOADED` large enough for the explicitly selected preparation models.
The command closes its owned models/cache and restores process environment overrides.

For deployment:

```text
VEYA_CACHE_PATH=/persistent/decisions.sqlite
VEYA_CACHE_REQUIRED=1
VEYA_CACHE_TTL=none
VEYA_CACHE_MAXSIZE=none
VEYA_CACHE_FINGERPRINT=support-policy-v1
VEYA_CACHE_EXPECTED_STORE_ID=UUID_FROM_PREPARATION
VEYA_REPLAY_MANIFEST=/persistent/support.replay.json
VEYA_PRELOAD=0
```

An explicit preload still loads models. Direct Router construction with opt-in replay
defers automatic profile preload and inference-only thread setup until a model is
actually needed. HTTP/MCP retain their configured preload behavior; set
`VEYA_PRELOAD=0` for model-free service startup. Their inference-only thread/device
setup is deferred until a model is needed.
Approved nonempty `agent_kwargs` are checked against the manifest without importing
the Agent implementation just to inspect its signature. Preparation validates those
options against the real constructor. Per-model calibration fitted at runtime must
be persisted and configured reproducibly: a cold miss with different calibration is
refused rather than silently committing decisions from another policy.

## Store identity and loss detection

Every built-in SQLite store has a UUID retained across closing, pruning and clearing.
`DecisionCache.store_id` reads it. `expected_store_id=...` implicitly requires an
existing database and checks its identity before any schema or journal writes.
`require_existing=True` requires a compatible existing decision table; an old valid
cache without an identity table can be explicitly adopted and assigned a UUID.
Missing files, empty/unrelated databases, replacement identities and malformed
identity metadata are refused. In-memory and custom stores do not support these guards.

Keep the expected identity outside the database's disposable directory. An intact
UUID does not detect individual rows erased by an operator, a permitted expiry,
eviction, or restoring an older backup. Preserve and back up the authoritative data.
Required mode does not reconstruct missing answers. See [retention and recovery
limits](consistency.md).

## Eligibility and compatibility

The optimized path requires an ordinary Router with exactly one built-in SQLite
DecisionCache and approved built-in Agent, ONNXAgent or native ONNX metadata.
Custom/global/per-call hooks, custom Router subclasses, callable routing hints and
callable retention policies or prediction-hook deadlines keep their ordinary execution path so their effects are
preserved. Long-document scans retain their existing scan-aware caching path.
Replacing an approved Agent or adding Agent-level hooks changes its frozen policy
and is refused. Custom agents can remain registered outside the approved model set.

Manifests bind the cache UUID and fingerprint, key/library/routing contract versions,
frozen Unicode-table digest, routing registry, effective construction options,
calibration and approved artifact stamps. Device/precision and resource profiles
are not final-decision identities. File stamps use size, modification time and inode,
matching the local checkpoint identity contract; they are not a cryptographic check
against an attacker who can forge filesystem metadata.

Preparing an ONNX manifest additionally requires `onnx>=1.20` (the lightweight
distribution's `replay` extra supplies it). Preparation parses the official ONNX
schema without loading external tensor data and covers graph, function, nested and
sparse tensor references. Inlined graphs can still allocate their protobuf data
during this administrator operation. Sidecar paths must be relative, remain within
the graph directory after resolution, and contain no traversal or URI components.
Graph symlinks retain the logical parent used by ONNX Runtime. Subsequent replay
uses the prepared file list and imports neither ONNX nor ONNX Runtime.

Re-export deliberately after changing approved source/configuration/calibration.
Keeping an explicit cache fingerprint retains prior compatible decisions; changing
that fingerprint requests reevaluation. Automatic fingerprints include the routing
contract. Existing explicit-policy keys remain compatible for unchanged routing.
Previously ambiguous language-sensitive entries made with a pinned model/task and
an explicit calibration language are not reused: those keys now include that
otherwise hidden language. This prevents one language's calibrated result from
being replayed as another's.

## Verification

Run `python tests/test_replay.py`. It covers cold single/batch replay, mixed misses,
policy and store identities, TTL/renewal, confidence presentation, metadata tampering,
checkpoint changes, concurrent namespace changes, relative paths, ONNX sidecars and
an independent process that refuses inference-library imports. Actual checkpoint
startup measurements are reported separately from the controlled unit fixtures.

## Recorded checkpoint comparison

The [benchmark harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_verified_replay.py)
compares the same retained decision using an existing English checkpoint on this
Windows workstation, CPU/eager/fp32 and four threads. Five alternating fresh-process
pairs produced identical answers and zero model forwards in both modes. Ordinary
cold replay still loads weights to reach its cache hook; verified replay attempted
no Torch, Transformers, ONNX Runtime, Tokenizers or ONNX imports.

| Metric, median | Ordinary durable hit | Verified replay |
|---|---:|---:|
| Fresh process wall time, including startup and cleanup | 6,765.27 ms | 227.22 ms |
| Cache/Router operation after required imports | 4,480.66 ms | 8.82 ms |
| Peak child working set | 1,939.8 MiB | 37.4 MiB |
| Warm hit, 500 requests | 59.5 microseconds | 375.8 microseconds |

The cold process comparison improved about 29.8 times in this run. Strict metadata
verification adds warm-hit work, so this feature is not a warm-lookup speed claim.
The small cold tail sample, filesystem caches and uncontrolled workstation load
limit generalization. This measures cached decisions rather than uncached inference
or model accuracy. [Raw samples, source hashes and checkpoint identities](https://github.com/tgundhus/Veya/blob/main/research/results/verified_replay_windows_20261105.json).
