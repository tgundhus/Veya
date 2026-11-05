---
title: Experimental HTTP microbatching
description: Opt-in bounded collection of compatible local HTTP single requests, with queue-inclusive measurements and conservative fallbacks.
---

# Experimental HTTP microbatching

The HTTP server can collect adjacent compatible `/v1/systemone` requests into a
shared `Router.predict_batch` call. This is opt-in and applies to single-request
HTTP calls only. The explicit `/v1/systemone/batch` endpoint and tournament mode
keep their existing serialized execution paths.

```text
LAYA_MICROBATCH_MAX_STATES=8
LAYA_MICROBATCH_WAIT_MS=1
```

`LAYA_MICROBATCH_MAX_STATES` defaults to `0` (disabled); an enabled value must be
between 2 and 64. `LAYA_MICROBATCH_WAIT_MS` defaults to one millisecond and must be
finite and nonnegative. Zero collects work already queued without an intentional
collection delay. Invalid settings fail startup. Existing deployments keep their
original single-request behavior when the feature is disabled.

## Boundaries and compatibility

Authentication, admission, body deadlines, byte/state/question limits, surrogate
checks and control validation run before work is queued. The queue has at most
`LAYA_MAX_CONCURRENT` entries independently of admission, so repeated client
cancellations cannot grow retained queued bodies without bound. Overflow and
shutdown return 503 with `Retry-After`.

Only adjacent requests with exactly matching normalized question order and all
prediction controls are collected together. In particular, confidence thresholds
are never borrowed from another caller. Input order and per-request output/error
mapping are retained. The same single-worker gate serializes combined inference,
explicit batches, tournaments and idle unloading.

Combined model calls use the exact built-in Router and Agent/ONNXAgent types and already resident checkpoints
whose actual token widths are known. Aggregate state/question/token cost bounds
each model pass using `LAYA_MAX_BATCH_TOKENS`. A cold checkpoint or unknown width
uses the original per-request prediction path. Non-cache custom hooks, custom
Router/Agent subclasses, custom attached agents and callable deployment routing hints also use individual
prediction, preserving their side effects. The built-in `DecisionCache` remains
supported, including first-stored-winner replay.

A compatible group containing exact duplicate state bytes and a built-in decision
cache executes individually. The first prediction populates the cache before
later repeats are looked up, avoiding redundant cold rows and retaining the cache's
existing claim and output semantics. Other compatible cache-backed states can
still share a model call.

An ordinary combined-call failure is retried as individual predictions so a bad
request does not turn its valid neighbours into 422 errors or expose their error
details. This retry is restricted to the built-in/cache-only hook path. Required
decision-cache errors are terminal and are not retried as alternative predictions.

Cancellation affects only the caller's future. A cancelled job that has not
entered the model is skipped. Cancellation cannot interrupt an already running
shared Torch/ONNX call; other callers still receive their answers and the worker
updates the idle clock when it completes. Application shutdown settles outstanding
futures before closing the worker/cache; a restarted lifespan creates a fresh queue.

Batch shape can change floating-point probabilities, especially at reduced
precision. Microbatching does not by itself guarantee decision invariance. Retain
the final decision under an explicit policy when repeatability is required, and
test numerical parity on your own checkpoints and representative borderline cases.

## Timing and tradeoffs

With this feature enabled, `Server-Timing` reports separate `inference` and `queue`
durations. Queue time starts after request parsing/validation and includes the
collection window and wait for the worker. `X-Inference-Time-Ms` remains model
worker time; for a shared model call it is the call's total duration for each
member. External client wall-clock latency also includes HTTP/JSON overhead.

Cache-hit requests currently join the collection window, so cache-heavy workloads
can be slower than the default path. Keep the feature disabled until a
queue-inclusive benchmark shows a useful tradeoff for the deployment. No automatic
profile enables it.

## Regression tests and benchmark

```bash
python -m pytest tests/test_microbatch.py -q
python research/scripts/bench_microbatch.py --out research/results/microbatch.json
python research/scripts/bench_microbatch.py --checkpoint /models/laya \
  --requests 16 --out research/results/microbatch_cpu.json
```

The default benchmark uses a controlled fixed-overhead agent to test the scheduler,
not to claim model throughput. It compares enabled/disabled runs in alternating
order, reporting queue-inclusive p50/p95/p99 latency, throughput, states entering
the Agent batch API, API calls, exact result equality, and cache statistics. The
real-checkpoint run additionally counts actual Torch model forward attempts and
rows using a model pre-hook; nested `system_one` to `predict_batch` calls count once.
Cases include unique requests, a nominal 70% prewarmed-repeat mix (actual fraction
recorded), identical bursts and mixed schemas/models/lengths. The optional
local-checkpoint run uses warmed English CPU fp32 inference without model
downloads. It omits the multi-model synthetic case. ASGI transport avoids TCP;
the small tail sample and uncontrolled workstation load limit interpretation.

## Recorded local comparison

The [English CPU checkpoint run](https://github.com/tgundhus/Veya/blob/main/research/results/microbatch_cpu_windows_20261105.json)
uses 16 requests, concurrency eight, one `noul` question, fp32/eager and four CPU
threads. Values below are medians of three paired runs on the same Windows
workstation. With 16 samples, nearest-rank p95 and p99 both select the maximum.

| Workload | Requests/s, disabled → enabled | p95 ms, disabled → enabled | Torch forwards, disabled → enabled |
| --- | --- | --- | --- |
| Unique states | 8.98 → 11.92 | 928.50 → 679.65 | 16 → 2 |
| Prewarmed repeats (13/16 requests) | 45.15 → 44.70 | 349.63 → 223.35 | 3 → 3 |
| Initially uncached identical burst | 142.74 → 157.71 | 106.22 → 96.37 | 1 → 1 |

Exact outputs matched in every paired run. The repeat-heavy trace had slightly
lower throughput with microbatching; batching is therefore still disabled by
default. These short local measurements establish scheduler behavior and a
checkpoint-specific comparison, rather than a hardware-independent speed claim.
The duplicate-state guard preserves the single computed row for a fresh identical
burst. The [controlled synthetic run](https://github.com/tgundhus/Veya/blob/main/research/results/microbatch_synthetic_windows_20261105.json)
also covers mixed question schemas, checkpoint choices and state lengths. Both
artifacts include queue-inclusive p50/p95/p99, raw runs, source hashes and settings.
