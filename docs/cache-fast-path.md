---
title: Verified HTTP cache replay
description: Serve exact retained decisions without loading model weights or joining the model collection queue, using approved local replay metadata.
---

# Verified HTTP cache replay

A configured [replay manifest](replay.md) lets `/v1/systemone` authorize exact
retained decisions before model scheduling. An authorized hit bypasses both the
microbatch collection window and the inference worker, including when that worker
is computing another request. Misses continue through the existing bounded
prediction path. The explicit batch endpoint uses the Router's normal batch API,
which also supports verified replay before loading a model.

This is opt-in through `LAYA_REPLAY_MANIFEST`. With no manifest, HTTP scheduling
keeps its existing behavior. `LAYA_HTTP_REPLAY_FAST_PATH` defaults to `1` when
a manifest is configured. Set it to `0` to retain the Router's model-free replay
while keeping HTTP calls on the ordinary scheduling path. Invalid boolean values
fail startup. Inference remains entirely local.

## Initialize and approve a policy

Follow the [manifest setup](replay.md) to initialize the intended checkpoints,
calibration and decision store. The manifest binds the serving configuration and
store identity. Using the deployment's checkpoint and cache environment, approve
the chosen checkpoints with:

```bash
python -m laya.replay --output /persistent/review-policy.json --models english
```

The command loads the listed checkpoints and prints the approved store identity.
On an already configured Python Router, the equivalent export is:

```python
approval = router.export_replay_manifest("/persistent/review-policy.json")
print(approval["store_id"])
```

Use an explicit checkpoint revision or local checkpoint directory as required by
the manifest exporter. Export does not populate new decisions: they are stored by
ordinary predictions under that approved policy.

For subsequent serving, preserve the same checkpoint and calibration settings and
configure:

```text
LAYA_CACHE_PATH=/persistent/decisions.sqlite
LAYA_CACHE_REQUIRED=1
LAYA_CACHE_REQUIRE_EXISTING=1
LAYA_CACHE_EXPECTED_STORE_ID=<the UUID printed during approval>
LAYA_CACHE_FINGERPRINT=review-policy-v1
LAYA_CACHE_TTL=none
LAYA_CACHE_MAXSIZE=none
LAYA_REPLAY_MANIFEST=/persistent/review-policy.json
LAYA_HTTP_REPLAY_FAST_PATH=1
LAYA_PRELOAD=0
```

`LAYA_PRELOAD=0` allows a process whose traffic is entirely retained hits to keep
weights unloaded. A miss still loads its routed checkpoint normally. If the store
or approved metadata is lost, corrupt or incompatible, required replay returns an
unavailable error rather than bypassing storage. Re-export deliberately after
changing the policy, checkpoint or relevant configuration.

`LAYA_REPLAY_MANIFEST` requires a persistent `LAYA_CACHE_PATH`; an in-memory cache
cannot match the manifest's store identity. The existing-store flag and expected
UUID are validated before SQLite can create
a replacement file. They require `LAYA_CACHE_PATH`; an in-memory cache cannot
provide a persistent store identity. An empty flag uses its disabled default;
invalid boolean spellings fail configuration. `LAYA_CACHE_EXPECTED_STORE_ID`
also implies that the identified SQLite store must already exist.

## Resource and correctness boundaries

Authentication, body deadlines, state/question limits and prediction-control
validation precede replay scheduling. A dedicated cache worker performs bounded
lookups outside the event loop. It drains currently queued requests without an
intentional collection delay, and authorizes each request separately through
`Router.try_replay`. It never reads private store internals or constructs a model.

The cache queue holds at most `LAYA_MAX_CONCURRENT` waiting jobs, independently of
the HTTP admission limit. At most 64 jobs are active in a drain. Cancelled waiting
jobs are skipped; cancelling an active caller cannot interrupt an already running
SQLite operation. Its operation remains bounded by the worker and queue, so
repeated cancellations cannot create an unbounded executor backlog. Shutdown
settles active and queued request futures, drains the worker, then closes an
owned cache. Restarting the application's lifespan creates fresh worker state.

One request's error does not replace another request's result. Required storage
failures remain terminal sanitized HTTP 503 responses with `Retry-After: 1`.
Retention renewal, expiry, authoritative decoding, fingerprint checks and
per-call confidence gates are supplied by the shared replay implementation.

Custom Router subclasses, custom hooks, unsupported replay policies and tournament
requests keep their established prediction path. The fast path never removes the
model worker's serialization around inference and idle unloading. Authenticated
health reports whether model-free replay is configured and the cache's store UUID;
anonymous liveness does not disclose policy configuration or filesystem paths.

With a Torch replay manifest and no resident model, device normalization and
thread-count validation do not import Torch. Torch's device parser and captured
thread settings are applied once, before the first model load. Simple malformed
device spellings and invalid thread counts fail startup; device types or indices
that Torch rejects or rewrites fail when inference first needs that runtime.
Changing environment values afterward does not change the captured setup.
Cold health reports the configured device or `auto` with `device_is_preference`
true, rather than importing Torch to probe hardware availability.

Startup checks manifest/store/configuration compatibility after installing the
cache and closes that cache if validation fails. Unsupported hook chains disable
the dedicated HTTP cache worker. Health's `model_free_replay.enabled` describes
configured metadata; `http_fast_path` also reflects current replay eligibility
and the HTTP switch. Changed or unavailable approved metadata cannot silently
authorize a replacement answer.

## Timing and qualification

Verified HTTP hits include `X-Decision-Cache: hit`. `Server-Timing` includes `cache`
for lookup work and `cache_queue` for its wait. Their `inference` duration and
`X-Inference-Time-Ms` are zero because no inference-worker call ran. A cache miss
includes lookup timings plus its usual inference and optional model-queue timing.

The separate cache probe adds work to misses. Fully queued cache-hit groups can
already complete quickly through ordinary microbatching. Choose this path using
representative traffic and queue-inclusive latency, particularly cache hits mixed
with slow model work. Use `LAYA_HTTP_REPLAY_FAST_PATH=0` when that tradeoff does
not help the deployment. It does not increase neural-model speed or make a retained
decision more accurate.

```bash
python -m pytest tests/test_http_cache_replay.py -q
python research/scripts/bench_http_replay.py --out research/results/http_replay.json
python research/scripts/bench_http_replay.py --checkpoint /models/english \
  --requests 16 --out research/results/http_replay_cpu.json
```

The benchmark alternates queued and fast-path serving with equivalent approved
policy/configuration metadata against fresh SQLite FULL stores. Each run exports
a manifest for that store's own UUID. It uses either a controlled fixed-overhead
fixture (requiring the ONNX extra for manifest export) or existing English CPU
fp32 weights. `LAYA_HTTP_REPLAY_FAST_PATH` is the scheduling difference between each pair.
It records all-hit, mixed-repeat and all-miss traces; raw
p50/p95/p99, throughput, cache/model counters, source/checkpoint identities and
response parity are retained. ASGI transport omits TCP, and client latency starts
after the client concurrency semaphore. Small samples and uncontrolled machine
load limit hardware-independent conclusions.

## Recorded local comparison

The [English CPU fp32 run](https://github.com/tgundhus/Veya/blob/main/research/results/http_replay_cpu_windows_20261105.json)
uses the existing English checkpoint, four CPU threads, one `noul` question,
16 requests and concurrency eight. Both modes approve the same policy/configuration,
with a fresh store UUID and manifest per run. The HTTP fast-path switch selects
the scheduling difference. Values are medians of three alternating
paired runs on this Windows workstation. With 16 requests, nearest-rank p95 and
p99 both select the maximum observation within each run.

| Workload | Requests/s, queued → fast | p50 ms, queued → fast | p95 ms, queued → fast | Actual model forwards, queued → fast |
|---|---|---|---|---|
| All retained hits | 1,288.06 → 1,279.00 | 4.64 → 4.87 | 6.10 → 6.40 | 0 → 0 |
| 75% prewarmed hits | 37.77 → 43.14 | 184.20 → 6.52 | 237.58 → 363.24 | 4 → 2 |
| All new states | 15.37 → 13.71 | 495.62 → 553.51 | 537.47 → 606.70 | 2 → 2 |

All-hit throughput was approximately unchanged in this small local sample.
Mixed traffic's retained-hit p95 fell from 237.58 to 7.42 ms and its throughput
improved about 14%, while its overall p95 worsened. Its remaining misses used
different model groups and client-concurrency timing; the run does not isolate
each contribution to that tail. All-new throughput decreased about 11%.
These tradeoffs support the independent opt-out for
cache-poor traffic; this path is not an automatic improvement for every workload.
The actual model row counts stayed at zero, four and 16 respectively in both
modes. Mixed traffic used four `[1, 43]` input-tensor forwards in queued mode
and two `[2, 43]` forwards in fast mode; all-new traffic used two `[8, 43]`
forwards in either mode. Every paired response payload matched exactly, measured probability drift
and label flips were zero, and cache error counts were zero.

The [controlled synthetic run](https://github.com/tgundhus/Veya/blob/main/research/results/http_replay_synthetic_windows_20261105.json)
uses 64 requests with the same concurrency and repeat count. Its mixed-case
throughput improved while all-new throughput decreased, consistent with the
purpose and cost of a separate cache probe. Both artifacts retain raw latencies,
63 source hashes, checkpoint identity where applicable, actual forward shapes,
per-request hit flags and input-ordered latency slices, and per-run counters.
They measure warmed resident serving, including worker startup, rather than
model loading or TCP transport. The checkpoint's load warning concerned a
clamped `choice:11+` temperature; this benchmark's only question is `noul`.
