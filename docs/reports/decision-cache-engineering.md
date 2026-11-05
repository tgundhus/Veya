---
title: Decision cache engineering
description: How DecisionCache keys, stores, expires and replays decisions, what each step costs, how it behaves under concurrency, and how one cache serves fp32, int8 and ONNX.
type: report
specificity: "laya/consistency.py as of commit 470f87e (Laya 0.3.20). Costs measured on a 4-core Intel Xeon at 2.8 GHz with Python 3.11 and SQLite from the standard library, 100,000 decisions held. The backend check ran on a synthetic english checkpoint: the published architecture with random weights."
credibility: "Costs come from research/scripts/bench_decision_cache.py (research/results/decision_cache_cpu_xeon4_20260926.json). Behaviour is pinned by tests/test_consistency.py (143 checks, stable over 20 repeated runs) and by the API contract in tests/test_hooks_api.py. Cross-backend replay was checked with research/scripts/check_backend_sharing.py. The Redis store was checked against fakeredis, not a live server."
---

# Decision cache engineering

`DecisionCache` is a prediction hook. Before inference it looks every state up, and a hit skips
the forward pass. After inference it stores what the model returned. User documentation is in
[Decision consistency](../consistency.md#replaying-decisions); this report records the design and
its measurements.

## Design

| part | choice | why |
|---|---|---|
| key | a 16-byte BLAKE2b hash of canonical JSON: key version, model fingerprint, checkpoint, route (only with per-language temperatures), token budget, questions in their given order, and the state | option order is positional, so it must be in the key; BLAKE2b was 4.9 µs against 8.2 µs for SHA-256 |
| fingerprint | model id, revision, calibration temperatures, configured token budget, Laya version | an upgrade starts fresh; `fingerprint="..."` pins it across upgrades |
| not in the key | device, precision, backend | a mixed fleet shares one set of decisions |
| value | compact JSON, zlib-compressed when smaller | 263-453 bytes become 179-182; no request text is stored |
| conflicts | the first decision stored wins, everywhere | every caller of a request gets one answer while its entry lives |
| expiry | per decision; `ttl` is seconds, None, or `ttl(questions, result)` | retention can depend on the question set or the answer |
| renewal | `renew_on_hit=True` renews at most every ttl/16 | a decision lives as long as it is used, and a hit almost never writes |
| stores | memory (LRU), SQLite, or any `DecisionStore` | a shared store spans machines without a hosted dependency |

## Costs

| step | cost |
|---|---|
| request key, 140 / 2,000 / 20,000-character state | 15.6 / 22.0 / 83.5 µs |
| encode a decision / decode it | 15-19 µs / 7-9 µs |
| memory store hit, and heap bytes per decision | 0.48 µs, 336 bytes |
| SQLite store hit, and disk bytes per decision | 7.5 µs, 319 bytes |
| SQLite store of one decision | 115 µs (276 µs with `synchronous=FULL`) |
| a whole `Router.predict` hit, 140 / 2,000 / 20,000 characters | 73 / 76 / 167 µs |
| extra cost of a miss with coalescing on | about 15 µs (93 → 111 µs over an instant fake model) |

A hit costs 0.015% of a 503 ms CPU request. Stores are synchronous. A background writer took
1.4 µs against 45 µs, but it would give up first-decision-wins across processes, and 45 µs is
0.15% of a single-question T4 request.

The Router's detection memo is what made hits cheap. Language detection was most of a hit's
cost, so the Router keeps the last 4,096 detections under a hash of the state. That took a
2,000-character hit from 314 to 76 µs.

## Concurrent identical requests

Within one process, a request that misses while another thread is computing it waits for that
decision instead of running the model. In the tests, eight threads sending one request together
over a 0.3 s model ran it once; the other seven were hits. A state repeated inside one call is
computed once and copied.

Deadlock freedom rests on four rules:

1. **Wait, then claim, atomically.** Under one lock, a request claims its missing keys only when
   no other thread owns any of them. Otherwise it releases the lock, waits and re-reads. An owner
   therefore never waits, and two requests can never wait on each other.
2. **Never wait where you could be waited on.** Nothing waits for its own batch, under
   `hooks_concurrent=False` (end hooks need the lock the waiter would hold) or under
   `hooks_timeout` (the hook runs on a helper thread).
3. **Always release.** The end hook releases claims in a `finally`, and so does a finalizer if
   a context is dropped without one. A failed forward pass releases its waiters at once.
4. **Bound a stuck owner.** A waiter gives up after five minutes, drops the stale claim and
   computes the request itself.

Across processes and machines there is no waiting; first-decision-wins still gives one answer.
`coalesce=False` turns waiting off, and `cache_info()["coalesced"]` counts the waits.

## One cache for every backend

Because the key has no device, precision or backend in it, a decision stored by the fp32
`Agent` is replayed by an int8 `Agent` or an `ONNXAgent` of the same checkpoint.
`research/scripts/check_backend_sharing.py` on the synthetic english checkpoint:

| backend | alone: same answer as fp32? | over one SQLite file |
|---|---|---|
| PyTorch fp32 | — | 589 ms, stored |
| PyTorch int8 | no | 0.48 ms, replayed fp32's decision |
| ONNX | no | 0.28 ms, replayed fp32's decision |

The model id or path is in the fingerprint, so load both backends the same way, or give both
caches the same `fingerprint="..."`. This makes int8 safe to deploy for repeats: its arithmetic
differs, but a request answered once keeps that answer on every backend.

## Sharing between machines

A `DecisionStore` needs seven methods: `get`, `add`, `renew`, `prune`, `__len__`, `clear` and
`close`, over bytes keys and values with expiries in Unix seconds. It must be thread-safe, and
its `add` must keep an existing unexpired value and return it. That one rule is what makes the
first decision win across machines.
[`examples/hooks/decision_store_redis.py`](https://github.com/tgundhus/Veya/blob/main/examples/hooks/decision_store_redis.py)
implements it with `SET ... NX` and Redis's own expiry. Laya itself ships no network store.

## Pre-warming

Known requests, such as a backlog, yesterday's traffic or an evaluation set, can be computed
ahead of time with `router.predict_batch` and the cache installed, for instance on a GPU.
Production machines on CPU then replay them, since the device is not in the key. The recipe is
in [Decision consistency](../consistency.md#pre-warming).

## Limits

- It matches exact requests only: one character of difference is a new request.
- It keeps a wrong answer until it expires. Pair it with `decision_margins` for review.
- On an `Agent` with per-language temperatures it stands aside, because a hook cannot see the
  call's `lang`. Install it on the `Router` there.

## Reproduce

```bash
python tests/test_consistency.py
python tests/test_hooks_api.py
python research/scripts/bench_decision_cache.py
python research/scripts/check_backend_sharing.py --model ID --onnx english.onnx
python examples/hooks/decision_store_redis.py      # needs redis and fakeredis, or REDIS_URL
```
