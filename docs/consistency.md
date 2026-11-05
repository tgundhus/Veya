---
title: Decision consistency
description: Why the same request can get a different answer, how close an answer is to flipping, and how to replay decisions so a repeated request gets the answer it got before.
type: explanation
specificity: "Laya 0.3.20. Consistency figures come from the archived Feishu run in research/benchmarks/feishu_zh (64 cases x 3 repeats: Jev 1.13.0 over its API, laya-multilingual on an Apple M4 in float32). Timings come from a 4-core Xeon CPU."
credibility: "Measured. tests/test_consistency.py recomputes every consistency figure from the archived raw responses, and research/scripts/bench_decision_cache.py and bench_stages.py reproduce the timings, and research/scripts/check_backend_sharing.py the fp32/int8/ONNX replay; the coalescing figures come from tests/test_consistency.py. The stage timings on this page used random weights with the published architectures, because the machine could not reach Hugging Face; they are right about where time goes and not about accuracy."
---

# Decision consistency

The figures on this page describe the archived 0.3.20 runs identified above. The
[October review](reports/production-review.md) records checks against the current fork and
clarifies local checkpoint identity and retention limits.

The same request can come back with a different answer for two unrelated reasons: the model
varies between runs, or something around the model changed. Laya's forward pass samples nothing,
so the first reason does not apply to it on a fixed setup, but the second one does. This page
shows how to see how close an answer is to flipping, and how to make a repeated request get the
answer it got the first time.

## What moves an answer

A hosted decision API is re-evaluated on every call. In the archived Feishu comparison, the same
64 requests were sent three times to Jev 1.13.0 and to `laya-multilingual`:

| identical across three runs | Jev 1.13.0 | Laya |
|---|---|---|
| whole response, probabilities and `confidence` included | 29 of 128 (23%) | 128 of 128 |
| `choice` probabilities | 38 of 64 | 64 of 64 |
| yes/no values of the four-question mode | 0 of 64 | 64 of 64 |
| the decision the benchmark acts on (label) | 127 of 128 | 128 of 128 |

Jev's yes/no values moved by a median of 0.01 and at most 0.08 between runs, and in 9 answers its
`confidence` changed while the probabilities it printed, to two decimals, did not. The one label
that changed (`cross_chat-03`: `todo`, `todo`, `noise`) was the least decided answer in the set:
0.53 against 0.44, then 0.51 against 0.44, then 0.45 against 0.50.

Laya returned the same bytes every time. What can still move one of its answers:

- **Hardware and precision.** CUDA runs in bf16 or fp16 autocast, and MPS in fp16 from five rows
  up, so the same request on another device can differ in the last digits.
- **Batch shape.** Padding a state next to longer ones changes floating-point summation order;
  `predict_batch(..., sort_by_length=True)` says so in its docstring.
- **The model.** A new revision, a refitted temperature, or a `max_len` change is a different
  model.
- **The request.** Option order is positional, so reordered options are a different request
  (Laya issue #166); on 20 options the answer changed
  for 15-23% of permutations ([option-order robustness](benchmarks.md)).

Each of these only flips an answer that sits close to its decision boundary.

## How close an answer is to flipping

`decision_margins` reads a result and returns, per question, how far the answer is from flipping,
from 0 on the boundary to 1:

```python
from laya import Router, decision_margins

router = Router()
result = router.predict(state, questions)
margins = decision_margins(result, noul_threshold={"urgent": 0.75})
borderline = [name for name, margin in margins.items() if margin < 0.1]
```

For `choice` and `score` the margin is the gap between the two most likely options. For `noul`
it is twice the distance from P(yes) to the threshold you act on (0.5 unless you say otherwise),
which at 0.5 is the gap between P(yes) and P(no). It needs nothing but the payload, so it reads a
Jev-compatible response the same way.

On the archive, three of Jev's answers changed between runs, and a margin under 0.1 marked all
three in every run, while marking 26 of the 960 answers it gave across the runs (2.7%). A model
that varies between runs does know when it is close. What it cannot know is which side the next
run lands on: each probability it reports is itself one draw.

For Laya the margin means something else, because its answers do not move between runs: it says
how likely the answer is to be wrong. On the same archive, which ran the base multilingual
checkpoint with no fine-tuning, 1 of the 14 `choice` answers with a margin under 0.05 was right,
against 14 of the 36 with a margin of 0.2 or more. Send low-margin answers to review or to a
fallback rather than acting on them.

## Replaying decisions

`DecisionCache` is a [prediction hook](hooks/index.md) that stores every decision under a hash of
the request and replays it when the same request comes again:

```python
from laya import DecisionCache, Router

cache = DecisionCache("decisions.sqlite", ttl=30 * 24 * 3600)  # or DecisionCache() for memory
router = Router(hooks=[cache])

router.predict(state, questions)   # runs the model and stores the decision
router.predict(state, questions)   # the stored decision, no forward pass
```

### Required decision storage

The default cache is an accelerator: store errors warn and allow inference to
continue. For a policy that must not return an uncommitted replacement answer,
opt into required storage:

```python
from laya import DecisionCache, DecisionCacheError, Router

cache = DecisionCache(
    "decisions.sqlite", ttl=None, maxsize=None,
    fingerprint="review-policy-v1", required=True,
)
router = Router(hooks=[cache])
try:
    result = router.predict(state, questions)
except DecisionCacheError:
    # Return an explicit unavailable/error state; do not silently evaluate again.
    result = None
```

Required mode raises `DecisionCacheError` when it cannot read/decode a stored
answer, serialize/retain/commit a new answer, confirm every written row, renew a
decision, or finish a coalesced wait. The error remains fatal with
`hooks_raise=False` and on any prediction-start/end hook deadline in its chain.
A timed-out Python hook cannot be
interrupted and may finish a commit later; that request still returns an error.
Do not retry by bypassing the required cache.

The required cache runs last in the prediction-start and prediction-end chains.
Other start hooks therefore finish rewriting the request before it is keyed;
other end hooks finish transforming a new result before it is committed. A hit
keeps the stored winning decision even if an end hook tries to change it. Fresh
commits also return decoded authoritative bytes, isolated from the model's mutable
result objects. An end hook cannot mark the context as failed to bypass storage.
Policy changes to those transformations need a new explicit fingerprint. Install
one required cache on the Router when using a Router; an Agent-level required
cache is supported for direct Agent calls only. Do not use another start hook's
`ctx.skip()` to answer ahead of it. Direct Agent calls with
per-language calibration must use a Router so language identity is visible.

For direct Agent/ONNXAgent `predict_long` calls, required storage preserves window
attribution and counts only windows actually inferred. Its end hooks must preserve
the scanned answer objects, prediction values and row order; reordering, replacing
or editing those answers raises `DecisionCacheError` before any new row is committed.
Annotation-only edits remain supported. This constraint prevents a transformed
answer from being retained under another window's identity. Existing replayed
answers always restore their stored winner.

Built-in SQLite uses WAL with `synchronous=FULL` in required mode (`NORMAL` stays
the default otherwise). Custom stores must implement atomic first-writer-wins
`add` correctly; successful acknowledgement and read-back cannot prove a remote
store's power-loss durability. An in-memory required cache still ends with its
process. Keep storage on a suitable persistent filesystem, with a backup/recovery
policy appropriate to the application.

`required=True` does not silently change retention. Finite TTLs, capacity limits,
explicit deletion and policy-version changes still permit reevaluation; use
`ttl=None` and `maxsize=None` when that is not wanted. A missing/deleted database
that is recreated empty is indistinguishable from a new policy store: required
mode cannot reconstruct an answer whose authoritative record has been lost.
Confidence/abstention flags are regenerated for the current call, and current
routing metadata is added as before. The retained decision is not a guarantee
that the original answer was correct.

- **Key.** A 16-byte BLAKE2b hash of the state, the questions in their given order, the per-call
  token budget, the checkpoint, and the model's fingerprint: its id, revision, calibration
  temperatures, configured token budget, subfolder and the Laya version. Unversioned local
  checkpoints also use artifact identity captured at load: verified digests when supplied,
  otherwise file size, modification time and inode. File metadata detects ordinary replacements;
  it does not verify contents or provide portable identity across machines. When the checkpoint has
  per-language temperatures the routing decision is part of the key too, because that is the one
  way the language reaches an answer. The device is not part of it, so a fleet of mixed hardware
  shares one set of decisions.
- **The first decision wins.** A request that missed at the same time as another, in another
  thread, process or machine sharing the store, returns the decision stored first instead of its
  own, and `cache_info()["conflicts"]` counts it. Every caller of a request gets one answer while
  its entry lives.
- **Retention.** Each decision carries its own expiry. `ttl` is the age in seconds after which a
  decision is answered afresh, `None` for never, or a function of the question set and the answer
  (below). `maxsize` (100,000 by default) is the most decisions the built-in stores hold; past it
  the least recently stored or renewed leave first, and `prune()` drops the expired ones on
  demand. Memory capacity is enforced on each write. SQLite checks the size bound periodically
  to amortize maintenance, so it can temporarily exceed `maxsize`, especially with several
  processes writing. Call `prune()` when a maintenance task must enforce the bound immediately.
- **Retention from last use.** With `renew_on_hit=True` a replay pushes the expiry out again, so
  a decision lasts as long as its request keeps coming and only one that stops coming expires.
  The renewal is written once the decision has aged a sixteenth of its lifetime, at most 16 times
  per lifetime, so a hit still almost never writes.
- **Upgrades.** A new revision or calibration changes the fingerprint and starts a fresh set of
  decisions. Pass `fingerprint="policy-v1"` to keep replaying the old ones across upgrades until
  they expire, when repeatability matters more than the new model.
- **What is stored.** Hashes and the answers, as compact JSON, compressed when that is smaller.
  No request text is written.

A `ttl` function sets retention per question set or per answer. It receives the questions and the
answer payload, and returns seconds, `None` to keep the decision for ever, or 0 not to keep it:

```python
from laya import DecisionCache, decision_margins

DAY = 24 * 3600

def retention(questions, result):
    if "fraud" in questions:
        return 7 * DAY                        # re-check fraud decisions weekly
    if min(decision_margins(result).values(), default=1.0) < 0.1:
        return 90 * DAY                       # close calls are the ones a recompute could flip
    return 30 * DAY

cache = DecisionCache("decisions.sqlite", ttl=retention, renew_on_hit=True)
```

### Sharing decisions between machines

The SQLite file is shared by the processes of one machine. To share one set of decisions across
machines, pass `store=` any object that follows `laya.DecisionStore`: `get`, `add`, `renew`,
`prune`, `__len__`, `clear` and `close`, over bytes keys and values with an expiry in Unix
seconds. The one rule a store must keep is in `add`: when a key already holds an unexpired
value, keep it and return it; that is what makes the first decision stored the one every machine
replays. Laya ships no network store itself, so it keeps depending on nothing hosted.
[`examples/hooks/decision_store_redis.py`](https://github.com/tgundhus/Veya/blob/main/examples/hooks/decision_store_redis.py)
is one over Redis, using `SET ... NX` for that rule and Redis's own expiry:

```python
import redis
from laya import DecisionCache, Router
from decision_store_redis import RedisDecisionStore

store = RedisDecisionStore(redis.Redis(host="cache.internal"))
router = Router(hooks=[DecisionCache(store=store, ttl=30 * 24 * 3600, renew_on_hit=True)])
```

Run the example file to check it against a Redis server (`REDIS_URL`) or fakeredis: two Routers,
standing for two machines, answer one request, and the second replays the first one's decision
without running its model.

A store that fails never fails a prediction. When a call raises (Redis cannot be reached, a
SQLite file is locked or its disk is full) or a stored value cannot be decoded, that request is
answered by the model as if there were no cache, and a value that cannot be decoded is replaced
by the new decision. The first failure warns, and `cache_info()["errors"]` counts them, so watch
it where decisions must stay consistent: while the store is down, repeats are recomputed.

### Concurrent identical requests

When the same request arrives on several threads at once, the first one to miss runs the model
and the others wait for its decision instead of computing their own. A burst of retries or a
fan-out then costs one forward pass. In the test suite, eight threads sending one request
together over a model that took 0.3 s ran it once; the other seven were hits, and
`cache_info()["coalesced"]` counts the ones that waited. A state repeated within one
`predict_batch` is computed once and copied to its repeats.

A request never waits on itself, so this cannot deadlock. It does not wait for the other states
of its own batch, under `hooks_concurrent=False` (hooks hold a lock there) or under
`hooks_timeout` (hooks run on a helper thread). An owner whose forward pass fails releases its
waiters at once, and they compute for themselves. One that hangs holds them for at most five
minutes, after which its claim is dropped. Different requests never wait for each other. Waiting
is per process; across processes and machines the first decision stored still wins.
`DecisionCache(coalesce=False)` turns waiting off. A miss costs about 15 µs more with it on, next
to hundreds of milliseconds for the forward pass it can save.

### One cache for every backend

The device, the precision and the runtime are not part of the key. A decision stored by the
PyTorch fp32 `Agent` is therefore replayed by an int8 `Agent` or an `ONNXAgent` of the same
checkpoint, as long as both are loaded by the same id or path, which is part of the fingerprint.
You can serve the fast int8 path and still give every repeated request the answer it got first,
whichever backend computed it. On the synthetic checkpoint, the three backends' own answers
differed, and over one SQLite file int8 and ONNX replayed the fp32 decision in 0.48 and 0.28 ms,
against 589 ms to compute it:

```bash
python research/scripts/check_backend_sharing.py --model convaiinnovations/laya --onnx english.onnx
```

Loading one backend from a local path and another from the Hub id gives two fingerprints. Pass
the same `fingerprint="..."` to both caches to share their decisions anyway.

### Pre-warming

Requests you know are coming, such as a backlog, yesterday's traffic or a fixed evaluation set,
can be answered before anyone asks. Run them through a Router with the cache installed, in
batches, on the fastest hardware you have; production then only replays:

```python
cache = DecisionCache("decisions.sqlite", ttl=30 * 24 * 3600, renew_on_hit=True)
router = Router(hooks=[cache])
for chunk in (requests[i:i + 64] for i in range(0, len(requests), 64)):
    router.predict_batch(chunk)   # [{"state": ..., "questions": ...}, ...]
```

Warm with the same questions, token budget and checkpoint that production will send, or the keys
will not match. A GPU machine can warm the file that CPU machines then serve from, since the
device is not part of the key.

A cache makes repeats identical; it does not make a borderline answer right, and it keeps a wrong
one until it expires. It matches exact requests only, so a state that differs by one character is
a new request. On an `Agent` with per-language temperatures it stands aside, because a hook
cannot see the `lang` a call passed; install it on the `Router` there.

## Voting

Asking a model several times and taking the majority reduces run-to-run variation; it does not
remove it. If one run flips with probability q, a three-run majority still flips with probability
3q² − 2q³: 26% for a q of 1/3. Averaging the probabilities of the runs is better than counting
votes, and gating on the margin keeps the cost down: on the archive, re-asking Jev only when the
margin is under 0.1 would have re-asked 2.7% of answers.

Re-running Laya on the same input returns the same bytes, so a vote over repeats is a vote of
identical ballots. A vote over *equivalent* requests is not: averaging the probabilities over
the rotations of a `choice` question's options, sent as one `predict_batch`, cancels the option
position bias above. Measure it on your own data before relying on it:

```python
def rotated_vote(router, state, name, question):
    options = list(question["criteria"].items())
    requests = [{"state": state, "questions": {name: dict(question, criteria=dict(options[i:] + options[:i]))}}
                for i in range(len(options))]
    runs = [r["answers"][name]["probabilities"] for r in router.predict_batch(requests)]
    mean = {label: sum(run[label] for run in runs) / len(runs) for label, _ in options}
    return max(mean, key=mean.get), mean
```

## Cost

A hit skips everything from tokenization to decoding. On a 4-core CPU a hit over an instant
model took 73, 76 and 167 µs for 140, 2,000 and 20,000-character states, against hundreds of
milliseconds to seconds through the model; the Router remembers the language detection of states
it has seen, which was most of a hit's cost before. Hashing the request costs 16-84 µs depending
on its length, a SQLite lookup 7.5 µs, and a stored decision takes about 320 bytes on disk.
Storing happens only on a miss and takes 45 µs, which is why it is not deferred to a background
thread. [Where a request's time goes](performance.md) has the full measurements.

## Reproduce

```bash
python tests/test_consistency.py                          # the archive figures above
python research/scripts/bench_decision_cache.py           # cache, routing and hook costs
python research/scripts/bench_stages.py --device mps      # stage timings on real checkpoints
```
