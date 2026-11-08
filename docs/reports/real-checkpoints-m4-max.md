---
title: Real checkpoints on an Apple M4 Max
description: The branch's consistency, decision-memory and speed claims re-measured on the published checkpoints against main, and the benchmarks the first round left out.
type: report
specificity: "Branch claude/blissful-keller-kuw2r7 at 8f4fd31 against main at 4066d5d; the merge section, branch merge/upstream-0.3.21 against upstream v0.3.21 (9d95567). The published checkpoints convaiinnovations/laya (english, revision 55cf4c4e) and its multilingual subfolder, in float32 unless a row says otherwise. Apple M4 Max (10 performance and 4 efficiency cores, 36 GB), macOS 26.3.1, Python 3.11.16, PyTorch 2.14.0 (10 threads), ONNX Runtime 1.30.0, SQLite 3.53.1. 841 requests: the 13 English scenarios, the 128 Feishu requests and 700 MASSIVE intent cases in 11 languages."
credibility: "Answers, agreement, accuracy, retention and restart results are counts, and do not depend on timing. Timings were taken while other work on the machine held the load average between 25 and 60 (headless browsers running other projects' test suites, which also used the GPU), so every latency comparison alternates the two sides and carries a 95% bootstrap interval. An A/A run of main against itself put the noise floor of separate-process comparisons at about 15%; the same-model test, which runs both sides around one model in one process, resolves new-request differences to 1-5%. No GPU and no x86 runs."
---

# Real checkpoints on an Apple M4 Max

The [first round](index.md) measured on a Xeon with random weights, because that machine could not
reach Hugging Face. This round reruns the claims on the published checkpoints, on the hardware the
consistency archive came from (Apple silicon), against `main`, and adds what the first round left
out: how exactly retention holds, what it buys, what survives a restart, what several processes and
threads do to the cache, how served traffic sees it, and which requests the cache cannot help.

## Verdicts

| area | original (`main`) | this branch | measured here | verdict |
|---|---|---|---|---|
| Answers and accuracy | — | identical | 841 of 841 requests byte-identical on CPU and on MPS; language detection identical on 179,214 inputs | **same** |
| Repeatability on one machine | identical for an identical call | identical | CPU and MPS repeat byte for byte. But on MPS, Laya's default device on a Mac, a request answered in a batch runs in fp16: 78% of payloads differ from the same request alone, and 4 of 1,076 decisions flip. The cache replays 841 of 841 | **better, with the cache**; the original is repeatable per call shape, not per request |
| Repeatability across machines, precision, upgrades | not guaranteed | first decision replayed | fp32 on CPU, MPS or ONNX: 0 decisions changed. fp16: 4 of 1,076. int8: 10-28%. Every backend replayed 841 of 841 through one cache; an upgrade starts fresh | **better** |
| Repeated request | full model run | replayed | a model run is 19-182 ms on MPS and 77-491 ms on CPU (a ticket with 1 question to an email with 3); a replay through the Router is 0.04-0.06 ms | **far better**: 590-11,400x |
| New request, default settings | — | a little less work | 0.1-0.8 ms less work per request (8.5 ms per batch of 32), 0.2-0.9% of a request; around one shared model, level on CPU and 1-4% faster on four of seven MPS requests | **same, slightly faster on MPS** |
| New request, opt-in int8 | — | 1.9-2.1x faster on an AVX-512 Xeon | on Apple silicon no faster (ONNX int8 1.02x) or slower (PyTorch int8 0.36x), and it changes 10% (ONNX) to 28% (PyTorch) of decisions, confident ones included; PyTorch int8 costs 10.5 points of accuracy on MASSIVE English | **worse here**; measure int8 on trained checkpoints per platform before recommending it |
| Knowing when an answer is shaky | — | `decision_margins` | a margin under 0.05 flags 2.5% of answers and catches all 4 fp16 flips; it catches only a few int8 flips | **better** for precision drift, not for quantization |
| Retention period | — | per decision, renewable | 0 disagreements with the documented rules in 348,004 checked requests, memory and SQLite | **correct** |
| Memory across restarts | — | SQLite or any store | a fresh process replayed 20,000 of 20,000 decisions byte for byte; nothing after a model upgrade or past the ttl | **works as documented** |
| Concurrent identical requests | each runs the model | one run shared | 16 threads, one new request: 1 forward pass in 0.22 s against 16 in 1.26 s | **better** |
| Served traffic | — | — | `laya-serve` has no setting for the cache; wired in by code, a hit waits behind other clients' misses (169 ms instead of 0.68 ms) | **gap** |

## Answers are unchanged

`research/scripts/bench_branch_ab.py` answers every request through a `Router` in two checkouts,
each in its own process, and compares the whole payload, `routing` included:

| device | requests byte-identical, main against branch | the same in a second fresh process |
|---|---|---|
| CPU | 841 of 841 | yes, both checkouts |
| MPS | 841 of 841 | yes, both checkouts |

Language detection feeds routing, which the branch sped up. `research/scripts/check_lang_parity.py`
runs `laya.lang.analyse` from both checkouts over every utterance of the MASSIVE test split in all
51 languages (151,674), 2,040 long texts of about 2,000 characters built from them, and 25,500 dict
states: **179,214 of 179,214 results identical**.

The speed-up is an English one. On English text the branch detects 3.2x faster at 2,000
characters (679 → 214 µs) and 1.7-1.9x on short text, as the first round reported. Over the other
50 languages it is 1.10-1.16x (2,267 → 2,054 µs on a 2,000-character text), because the fast
paths are ASCII paths.

## Consistency across setups

`research/scripts/bench_backend_consistency.py` answers all 841 requests on each setup and compares
it with PyTorch fp32 on CPU answering each request on its own. The Router sent 306 requests (349
answers) to the English checkpoint and 535 (727 answers) to the multilingual one.

| setup | `answers` identical | decisions changed | largest probability change |
|---|---|---|---|
| CPU fp32 again | 841 of 841 | 0 of 1,076 | 0 |
| CPU fp32 in a batch of five (padding) | 829 of 841 | 0 | 0.0001 |
| ONNX fp32 (English) | 296 of 306 | 0 of 349 | 0.0001 |
| MPS fp32 | 812 of 841 | 0 | 0.0001 |
| MPS, default policy, alone | 810 of 841 | 0 | 0.0031 |
| MPS, default policy, in a batch of five (fp16) | 189 of 841 | **4** (margins 0.014-0.023) | 0.10 |
| MPS fp16 always | 189 of 841 | 4 | 0.10 |
| ONNX int8 (English), as `export_onnx.py --int8` writes it | 99 of 306 | **35 of 349** (margins up to 0.994) | 0.9998 |
| ONNX int8 (English), per-channel weights | 109 of 306 | **10 of 349** (margins up to 0.998) | 0.91 |
| PyTorch int8, `qnnpack` | 72 of 841 | **299** (margins up to 1.0) | 1.0 |
| **any of the above behind one `DecisionCache`** | **841 of 841 replayed** | **0** | 0 |

What this says:

- **fp32 is fp32.** Another device, ONNX Runtime or padding moves a probability in the fourth
  decimal and changes no decision.
- **MPS batching changes answers on one machine.** Laya autocasts MPS to fp16 from
  `mps_amp_min_rows` rows (5), so the same request runs in fp32 alone and in fp16 inside a
  `predict_batch` or with five or more questions. That flipped 4 of 1,076 decisions, all within
  0.023 of their boundary. The original's "identical on one machine" holds for an identical call,
  not for a request that arrives in different company. A cache replays it either way.
- **int8 is not a precision change, it is a different model.** It changed decisions fp32 was sure
  of: 36% of the English checkpoint's answers under PyTorch and 10% under ONNX. On the 200 MASSIVE
  English cases (20 options), accuracy was 79.5% in fp32, 80.0% in ONNX int8 and **69.0% in
  PyTorch int8**. The multilingual checkpoint kept its accuracy under PyTorch int8 (55.8% → 56.5%
  on 407 MASSIVE cases) while changing 131 of those 407 decisions. The first round's "every
  changed answer had an fp32 margin of at most 0.0006" was an artefact of random weights.
- **Per-channel weights cut int8's damage by 70%.** Quantizing the ONNX weights per channel
  instead of per tensor changed 10 decisions instead of 35, at the same file size (581 MB) and
  accuracy. A few confident answers still flip. Laya 0.3.21's `--quantize` does exactly this, and
  since the merge `scripts/export_onnx.py` uses it (`--int8` is kept as another name for it).

### Margins as a review filter

Reference margins of the changed decisions, and how many answers each threshold sends to review:

| margin under | answers flagged, of 1,076 | fp16 flips caught, of 4 | PyTorch int8 flips caught, of 299 |
|---|---|---|---|
| 0.02 | 10 (0.9%) | 3 | 7 |
| 0.05 | 27 (2.5%) | **4** | 18 |
| 0.1 | 53 (4.9%) | 4 | 35 |

On the English checkpoint alone the same thresholds flag 1, 4 and 9 of 349 answers and catch 1, 4
and 7 of ONNX int8's 35 flips. A margin under 0.05 is a good filter for what precision and
hardware do, in line with the 2.7% the first round measured on Jev. It is no protection against
int8.

## Memory of decisions

### Retention is exact

`research/scripts/bench_retention.py` drives the cache's hooks over an instant model with a
simulated clock, and checks every request against a reference model of the documented rules:

| rule checked | requests | disagreements (memory / SQLite) |
|---|---|---|
| fixed ttl of 1 hour, and of 30 days | 80,000 per store | 0 / 0 |
| ttl 1 hour with `renew_on_hit` (expiry moved when it would move by more than ttl/16) | 40,000 per store | 0 / 0 |
| a ttl function: 7 days for one question set, 1 hour for another (5 minutes when the answer's margin is under 0.1), 0 for a third, none for a fourth | 40,000 per store | 0 / 0 |
| `maxsize=1000` with no ttl: only the newest decisions are held | 14,002 stored per store | 0 / 0 |

Total: 348,004 requests, 0 disagreements, no decision replayed other than the one stored. A hot
request renewed 15.7 times per ttl, under the ceiling of 16. Two details worth knowing:

- **Renewal writes on most hits of a request that repeats less often than ttl/16.** With a 1-hour
  ttl and each request coming back about every 13 minutes, there were 0.52 writes per hit. On
  SQLite that turns a hit from a read into a write (48.5 µs median against 19.1 µs).
- **SQLite's bound is loose by design.** It is enforced every `min(1000, maxsize // 10)` stores,
  so `maxsize=1000` held up to 1,099.

### What a retention period buys

A month of traffic through each setting: 300,000 requests, 60,000 recurring ones with Zipf
popularity (s = 1) and 30% seen once, 121,837 distinct. "Repeats given their earlier decision" is
the share of requests seen before that got the decision their request got before; a re-decision
is a repeat computed again, a chance for its answer to change.

| retention | replayed | repeats given their earlier decision | re-decisions | decisions held at most | store writes |
|---|---|---|---|---|---|
| ttl 1 hour | 23.1% | 38.9% | 108,804 | 3,546 | 230,641 |
| ttl 1 hour, renew on hit | 26.5% | 44.6% | 98,664 | 3,715 | 276,144 |
| ttl 1 day | 42.0% | 70.8% | 52,040 | 9,581 | 173,877 |
| ttl 1 day, renew on hit | 45.4% | 76.5% | 41,907 | 9,987 | 223,357 |
| ttl 7 days | 52.9% | 89.0% | 19,504 | 40,392 | 141,341 |
| ttl 7 days, renew on hit | 55.7% | 93.7% | 11,191 | 41,821 | 188,560 |
| ttl 30 days, or none | 58.3% | 98.2% | 3,131 | 100,000 (the default bound) | 124,968 |
| ttl 30 days, renew on hit | 59.3% | 99.8% | 334 | 100,000 | 162,874 |
| ttl 7 days, renew, `maxsize=10,000` | 48.5% | 81.7% | 32,672 | 10,000 | 188,560 |
| ttl 7 days, renew, in memory, restarted daily | 39.6% | 66.7% | 59,317 | 6,162 | 192,337 |
| ttl 7 days, renew, SQLite (survives restarts) | 55.7% | 93.7% | 11,191 | 35,241 | 188,560 |

- Most of the benefit comes by a week: 89-94% of repeats keep their decision, against 98-99.8% at
  a month. Renewal adds 2-6 points for 20-33% more writes.
- The default `maxsize` of 100,000 is what binds at a month on this traffic: 100,000 decisions are
  28-37 MB in memory (below). Size it to the distinct requests a ttl spans.
- **An in-memory cache forgets at every restart.** Restarted daily, it gave 66.7% of repeats their
  decision instead of 93.7%. Use a SQLite file, or a shared store, wherever a deploy restarts the
  process.

### Restarts and upgrades

Four processes in turn over one SQLite file of 20,000 decisions (ttl 30 days):

| process | replayed | computed |
|---|---|---|
| 1. first run | 0 | 20,000 |
| 2. fresh process, same model | **20,000, byte for byte** (median 24.7 µs a request) | 0 |
| 3. fresh process, new model fingerprint | 0 | 20,000 |
| 4. fresh process 31 days later, same model | 0 (expired) | 20,000 |

### Several processes and threads

`research/scripts/bench_store_scaling.py`, through the cache's hooks over an instant model:

- **First decision wins across processes.** 2, 4 and 8 processes each computed their own decision
  (tagged with the process) for the same 2,000 requests, in shuffled order, at the same moment,
  over one SQLite file. Every process got the same decision for every request. With 8 processes,
  16,000 requests ran 2,027 forward passes; 27 of those raced another process and replayed its
  decision instead.
- **Processes share one file well, with a long tail.** 90% repeats and 10% new requests: 33,460
  requests a second from one process, 56,237 from eight, with p99 rising from 0.11 to 1.7 ms and
  the worst request waiting 486 ms on SQLite's write lock.
- **Threads do not add hits.** Hits a second through the hooks from 1, 2, 4, 8 and 16 threads of
  one process: 56,690, 37,648, 33,738, 34,168, 35,042 in memory; 47,566, 21,768, 18,904, 18,684,
  17,918 on SQLite. Contention on the store's lock and the GIL costs a third in memory and more
  than half on SQLite. It is far above what a model serves, so it is a ceiling to know, not a
  bottleneck.

### What the stores cost as they grow

Every single store call, 20,000 of each, at three sizes (`tail` section):

| held decisions | store | hit, median / max | store a decision, median / p99.9 / max |
|---|---|---|---|
| 10,000 | memory | 0.4 µs / 25 µs | 0.6 µs / 3 µs / 2.7 ms |
| 100,000 | memory | 0.6 µs / 201 µs | 0.6 µs / 2 µs / 17 ms |
| 1,000,000 | memory | 0.8 µs / 108 µs | 0.7 µs / 4 µs / 0.08 ms |
| 10,000 | SQLite, no bound | 5.0 µs / 1.3 ms | 23 µs / 5.6 ms / 19 ms |
| 100,000 | SQLite, no bound | 5.8 µs / 83 ms | 25 µs / 20 ms / 105 ms |
| 1,000,000 | SQLite, bounded at 1M | 7.3 µs / 1.6 ms | 28 µs / 57 ms / 79 ms |

- Medians barely move from 10,000 to 1,000,000 decisions.
- **SQLite's tail is tens of milliseconds, on the request path.** A bounded store counts its rows
  and prunes every 1,000 stores, and WAL checkpoints add their own pauses; a hit that arrives
  meanwhile waits for the store's lock (83 ms once at 100,000). The memory store's sweep for
  expired decisions is the pause in its column.
- **Both maintenance passes grow with the store and block every call while they run.** Timed on
  their own (`maintenance` section):

  | held decisions | memory: one sweep | how often | SQLite, bounded: one prune | how often |
  |---|---|---|---|---|
  | 10,000 | 0.8 ms | every 4,096 stores | 2.0 ms | every 1,000 stores |
  | 100,000 | 9.4 ms | every 25,000 | 8.0 ms | every 1,000 |
  | 1,000,000 | 161 ms | every 250,000 | 44 ms | every 1,000 |
  | 5,000,000 | 1,449 ms | every 1,250,000 | 82 ms | every 1,000 |

  At the default bound of 100,000 these are single-digit milliseconds. A store of millions of
  decisions stalls for a tenth of a second to over a second. Expiring incrementally (a slice per
  store, or an index by expiry) and keeping a row count instead of `COUNT(*)` would flatten both.
- **Renewal costs a write on the hit path.** A SQLite hit that renews took 48.5 µs against 19.1
  µs (p99.9 1.56 ms against 0.19 ms).
- **Size per decision** (real payloads, 1 to 10 answers): 298-385 bytes in memory, so 28-37 MB
  for the default 100,000; 305-423 bytes on disk, so 290-400 MB per million.
- The branch's other caches are small: the Router's memory of 4,096 language detections holds 2.6
  MB when full, and the question-head cache 8.5 MB per tokenizer at its bound of 1,024 twenty-option
  questions.

## Speed

### Where a request's time goes on this machine

`research/scripts/bench_stages.py` on the real checkpoints, median of 10 runs:

| request | MPS | of which encoder | CPU, 10 threads | of which encoder | a `DecisionCache` hit |
|---|---|---|---|---|---|
| English ticket, 1 question | 18.8 ms | 74% | 76.6 ms | 92% | |
| English ticket, 3 questions | 34.1 ms | 82% | 143.9 ms | 93% | 0.058 ms (MPS), 0.037 ms (CPU) |
| English ticket, 10 questions | 135.5 ms | 89% | 308.6 ms | 93% | |
| English email, 3 questions | 182.0 ms | 89% | 490.8 ms | 92% | 0.063 ms (MPS), 0.043 ms (CPU) |
| batch of 32 tickets, 3 questions | 927.7 ms | 92% | 2,664 ms | 92% | |
| multilingual ticket, 3 questions | 26.7 ms | 74% | 70.5 ms | 89% | |
| multilingual email, 3 questions | 38.2 ms | 77% | 137.4 ms | 88% | |

- **A repeat is 590-2,900 times faster on MPS and 3,900-11,400 times faster on CPU.** A hit
  through the Router costs 37-63 µs here, against 73-76 µs on the Xeon.
- **On MPS the work around the model matters more, and has moved.** After the branch's fixes,
  routing and tokenization take 0.2-0.3 ms of a small request. The largest cost outside the
  model is now copying results back to the host, 1.9-2.2 ms (6-10% of a small request), then the
  decision head at 3-5 ms. On CPU everything outside encoder and head is 0.3-1% of a request.
- **fp16 on MPS only pays in bulk.** Forcing fp16 on a 3-question ticket made it 0.79x as fast;
  Laya's threshold of 5 rows is the right default. It is also what makes batching change answers
  (see [consistency](#consistency-across-setups)).

### New requests: no slower, slightly faster on MPS

The branch changed the code around the model, not the model. Timed apart from the model, over
alternating processes (`bench_branch_ab.py`, 4 rounds each):

| work around the model | main | branch | saved |
|---|---|---|---|
| English ticket, 3 questions: routing, then tokenizing state and rows | 0.036 + 0.332 ms | 0.024 + 0.059 ms | 0.29 ms |
| English ticket, 10 questions | 0.036 + 0.934 ms | 0.024 + 0.104 ms | 0.84 ms |
| English email, 3 questions | 0.565 + 0.591 ms | 0.365 + 0.333 ms | 0.46 ms |
| batch of 32 tickets, 3 questions | 1.157 + 10.235 ms | 0.768 + 2.136 ms | 8.49 ms |
| multilingual ticket, 3 questions | 0.037 + 0.276 ms | 0.024 + 0.058 ms | 0.23 ms |

That is 0.2-0.9% of a request on either device. End to end it is below what this machine could
resolve: an A/A run of `main` against itself reported "speed-ups" up to 1.17x, with intervals
excluding 1 when samples are resampled one by one, so only differences well past that mean
anything. Separate A/B runs of `main` against the branch came out between 0.87x and 1.11x per
scenario, in both directions. The fair test puts both request paths around one model in one
process (`bench_branch_same_model.py`): `main`'s package is imported under another name, both
Routers share the same model object, and requests alternate in adjacent pairs, so every pair sees
the same load. Median of 60 per-pair ratios (above 1: the branch is faster), with 95% intervals:

| request | CPU | MPS |
|---|---|---|
| English ticket, 1 question | 0.994 [0.970, 1.023] | 0.999 [0.994, 1.012] |
| English ticket, 3 questions | 0.999 [0.976, 1.044] | **1.011** [1.005, 1.018] |
| English ticket, 10 questions | 0.991 [0.954, 1.055] | **1.016** [1.011, 1.021] |
| English email, 3 questions | 1.016 [1.002, 1.030] | 1.005 [0.997, 1.014] |
| batch of 32 tickets, 3 questions | 1.002 [0.992, 1.011] | 1.001 [0.998, 1.005] |
| multilingual ticket, 3 questions | 1.035 [0.979, 1.101] | **1.042** [1.025, 1.055] |
| multilingual email, 3 questions | 0.997 [0.969, 1.034] | **1.030** [1.006, 1.043] |

Answers were identical in every pair. On CPU the branch is no slower on a new request, to within
2-5%. On MPS, where the model is fast enough for the work around it to show, it is 1-4% faster on
four of the seven requests (0.5-1.7 ms a request) and level on the rest.

### int8 and ONNX on Apple silicon

`research/scripts/bench_cpu_fast_path.py` on the 13 English requests, round-robin over 5 repeats:

| backend | ticket, 3 questions | email, 3 questions | total | geometric mean | same decision as fp32 | first round, Xeon with random weights |
|---|---|---|---|---|---|---|
| PyTorch int8 (`qnnpack`) | 0.45x | 0.35x | **0.36x** | 0.40x | 48 of 56 | 1.89x, 44 of 56 |
| ONNX fp32 | 1.45x | 1.00x | 1.07x | 1.22x | 56 of 56 | 1.06x, 56 of 56 |
| ONNX int8 | 1.54x | 1.09x | 1.02x | 1.22x | 53 of 56 | 2.07x, 28 of 56 |

- **int8 buys nothing on Apple silicon.** ONNX int8 runs at ONNX fp32's speed, and PyTorch's
  `qnnpack` int8 is 2.8x slower than fp32. The first round's 1.9-2.1x is an AVX-512 VNNI result.
- **ONNX fp32 is the faster CPU path for short requests** (2.4x on a 52-token ticket, 1.45x on
  164 tokens) and loses on long ones (0.94x at 2,766 tokens), as on the Xeon.
- **MPS beats every CPU option** on the requests timed both ways: 34 ms for the 3-question ticket
  against 105 ms for the fastest CPU backend (ONNX int8), and 182 ms for the email against 482 ms.
- On Apple silicon PyTorch selects no quantized engine, so `bench_stages.py --variants int8` and
  `check_backend_sharing.py` fail with `NoQEngine`, and `bench_cpu_fast_path.py` skips its int8
  backend. Setting `torch.backends.quantized.engine = "qnnpack"` first makes them run.

### Served traffic

`research/scripts/bench_cache_serving.py` on MPS.

**A stream of requests.** 600 requests drawn with Zipf popularity (s = 1) from 341 requests (the 13
English scenarios, 200 MASSIVE English cases and the 128 Feishu requests): 177 distinct, so 423
repeats, served one after another.

| setup | requests a second | median | p99 | repeats given their first answer |
|---|---|---|---|---|
| no cache (the default path) | 16.8 | 53.0 ms | 144 ms | 423 of 423 |
| memory `DecisionCache` | **61.0** | 0.80 ms | 93 ms | 423 of 423 |
| SQLite `DecisionCache` | **60.2** | 0.87 ms | 92 ms | 423 of 423 |

With 70% repeats the cache serves 3.6x the requests. Every repeat got its first answer without a
cache too, because each request here is answered alone on one device; batching or a second device
is what breaks that (see [consistency](#consistency-across-setups)). A hit in this stream costs
more than the 0.06 ms of back-to-back hits because each one follows a model run: on this loaded
machine a hit straight after an MPS miss took 0.22 ms, and 0.33 ms after a 50 ms pause.

**Concurrent identical requests.** K threads send the same new request at once (on CPU: see
below for why not MPS):

| threads | coalescing on: forward passes, time until all answered | off | answers identical |
|---|---|---|---|
| 2 | 1, 0.20 s | 2, 0.19 s | yes, both |
| 4 | 1, 0.24 s | 4, 0.43 s | yes, both |
| 8 | 1, 0.21 s | 8, 0.63 s | yes, both |
| 16 | 1, 0.22 s | 16, 1.26 s | yes, both |

## Serving gaps

These are not regressions (`main` has no cache at all), but they decide what the cache does in a
deployment:

- **`laya-serve` cannot turn the cache on.** It builds its Router from `LAYA_*` environment
  variables, and none installs a `DecisionCache` or sets its path, ttl or bound. Only code that
  builds the app itself (`create_app(router=Router(hooks=[DecisionCache(...)]))`) gets one. The
  MCP server has no setting for it either, and the Docker HTTP service runs `laya-serve`.
- **In the server, a hit waits for other clients' misses.** `create_app` runs every request,
  hits included, one at a time behind one lock on a one-thread executor. With a cache wired in:

  | request over HTTP (in process, MPS) | median | p99 |
  |---|---|---|
  | a hit, alone | 0.68 ms | 2.09 ms |
  | a miss, alone | 34.9 ms | 43.1 ms |
  | **a hit while 3 other clients send misses** | **169 ms** | **212 ms** |

  A 0.7 ms answer becomes a 170 ms one: the hit queues behind every miss ahead of it. Coalescing
  never happens either, since two requests are never in flight together. Looking the request up
  before taking the lock would keep hits at their own cost.
- **Two threads predicting on MPS at once abort the process** (SIGABRT from a Metal assertion,
  `failed assertion _status < MTLCommandBufferStatusCommitted`), reproduced with two threads and
  two different requests. PyTorch's MPS backend is not safe for concurrent forward passes; the
  server's single inference thread is what protects it, and any threaded caller of a `Router` on
  MPS needs the same. Coalescing makes concurrent *identical* requests safe there, since only one
  of them runs the model, but not different ones.

## Requests that mean the same

A `DecisionCache` replays exact repeats. `research/scripts/bench_equivalent_requests.py` asks each
request again in a form that means the same but is a new key, on the same device and model, and
counts the decisions that change:

| variation | English checkpoint | multilingual checkpoint | margin of the changed answers (median) |
|---|---|---|---|
| a choice question's options in reverse order | 67 of 348 (19%) | **171 of 471 (36%)** | 0.66-0.85 |
| the first option moved to the end | 31 of 348 (9%) | 102 of 471 (22%) | 0.34-0.76 |
| every space doubled | 73 of 343 (21%) | 72 of 298 (24%) | 0.56-0.79 |
| one trailing space | 5 of 349 (1.4%) | 48 of 727 (6.6%) | 0.15-0.18 |
| lower-cased | 3 of 56 (5%) | 15 of 320 (5%) | 0.07-0.14 |
| *for comparison: MPS fp16 instead of CPU fp32* | *1 of 349* | *3 of 727* | *0.02* |

This is the largest source of changing answers measured here, far larger than hardware or
precision, and the cache cannot help: each variant is a new request. Nearly all of it is in long
option lists (the 20-option MASSIVE questions) and hard questions (Feishu, where fp32 is right
31% of the time); the 3- to 5-option questions of the English scenarios did not flip on reordering.
The flips are confident ones, so margins do not flag them either. Laya 0.3.21's new
`usage["options"]` report shows part of why: on 147 of the 841 requests, every one a 20-option
MASSIVE question, the head budget leaves fewer distinct option spans than options (18 of 20), so
which options merge depends on their order.

Two things would remove most of it:

- **Send options in one fixed order.** Option order is positional, so Laya cannot reorder them for
  the caller without changing the answer. A client that builds its option list from a set or a
  database query should sort it. Where the order cannot be fixed, averaging over rotations of the
  options, the recipe in [Decision consistency](../consistency.md#voting), trades k forward passes
  for an order-free answer.
- **Normalise whitespace before the model and the key.** A hook placed before the cache that
  collapses runs of whitespace and strips the ends of string states makes the model see, and the
  cache key, one canonical text. Trailing spaces and doubled spaces then become exact repeats.

## Merging Laya 0.3.21

Veya then merged the original's release
0.3.21 (114 pull requests). Where
upstream had changed what the fork changed, the merged code was measured against Laya 0.3.21 as
released, on the same machine (`research/results/upstream0321_*_m4max_20260929.json`):

| check | result |
|---|---|
| Answers against 0.3.21, 841 requests | byte-identical on CPU and on MPS: 841 of 841 |
| Language detection against 0.3.21, 179,214 inputs (upstream's #531 and the fork's shortcuts together) | identical, and 1.11-1.20x faster |
| Tokenizing the state and question rows (0.3.21 reuses them within a call, #615; the fork keeps the question head across calls) | 1.4-6.8x faster on CPU: 0.83 → 0.12 ms for 10 questions, 9.3 → 2.2 ms for a batch of 32 |
| A new request around one model, against 0.3.21 | MPS: 0.4-2.7% faster in all seven requests, five with intervals clear of 1. CPU: level; a batch of 32 is 1% faster |
| MPS consistency after #559 changed the decision head's attention | unchanged: 0 decisions changed in fp32, the same 4 of 1,076 in fp16, 841 of 841 replayed |
| Upstream's per-channel int8 export (#498) | 10 of 349 decisions changed, as the per-channel test above found, against 35 per tensor; accuracy 64.5%; 600 MB |
| A `DecisionCache` hit after upstream's hook changes | 31 µs in memory, 36 µs on SQLite: no slower |
| Answers against Veya before the merge (MPS) | 694 of 841 byte-identical. The other 147 differ only by 0.3.21's new `usage["options"]` report, which says that a 20-option MASSIVE question kept 18 distinct option spans under the head budget. No probability changed. |

The merge needed four fixes, each with tests:

- **`build_sequence` would have failed every prediction.** Git kept upstream's new `return_stats`
  parameter but the fork's cached body, which ignored it, while `Agent` and `ONNXAgent` ask for
  three values. The cached question head now carries upstream's option statistics.
- **Abstention marks.** `min_confidence` marks answers before end hooks run, so the cache stored one
  call's `low_confidence` marks and replayed them to calls with another threshold or none, and a
  partial hit's replays went unmarked. The cache now stores answers unmarked, and `Agent`,
  `ONNXAgent` and `Router` mark again after their end hooks.
- **`Agent.predict_long` and `ONNXAgent.predict_long`** raised `ValueError` once every window of a
  document was cached, and lost the deciding window's attribution when some were. A cache answering
  one result per window is now aggregated as the scan it is.
- **`Router.predict_long`** and `Router.predict` shared a cache entry for the same state, so each
  replayed the other's answer. `PredictContext.scan` now tells the cache which it is answering.

A PR-Agent review of the merge found the `ONNXAgent` half of those two fixes missing, and four
faults in code that came from upstream. Each is now fixed, with a test that fails without the fix:

- `Router` dropped an `expected_sha256` pin passed in `agent_kwargs` whenever its own digest map
  had an empty entry for the checkpoint, including the `{}` that a model-keyed digest map in the
  environment gives every model it leaves out, and so loaded the weights unverified. An empty entry
  no longer removes a pin.
- `Router.predict_batch` retried an agent that predates both `lang` and `sort_by_length` once,
  dropping one of them, and then raised on the other.
- The LangChain runnables' `abatch` raised `IndexError` on an empty batch that came with an empty
  list of configs.
- `ONNXAgent`'s option-overflow error printed the already-truncated sequence length as what the
  question needed ("need 40 tokens, more than max_len=40"). It now says how many options fit.

The review's other findings are not bugs. The guardrail's score probabilities are keyed by level
index on both backends; `normalise_name` passes any value through `str()`; and `Agent` lowercases
noul criteria keys. The rest are upstream's choices, kept as released:

- `ONNXAgent`'s new `token` parameter comes before `subfolder`, as it does on `Agent`.
- `LayaGuardrail(threshold=)` is now a probability.
- Batch MCP requests check `task` only as a string.

Upstream's `test_question_token_reuse` counts tokenizer calls for its reuse within one call, so it
runs with the fork's cache across calls switched off, and a new test checks the two together. The
version moved to 0.3.21, which is part of the cache's fingerprint: decisions stored before the
merge are not replayed after it.

## What this changes in the other reports

- **int8.** [The overview](index.md) and [the performance report](request-performance.md) say
  int8 roughly halves CPU time (ONNX 2.07x, PyTorch 1.89x). That is an AVX-512 VNNI result: on
  Apple silicon ONNX int8 matches ONNX fp32 and PyTorch int8 is 2.8x slower. They also say that
  trained checkpoints "will change far fewer" answers than random weights did. On the trained
  English checkpoint int8 changed 10% (ONNX) and 36% (PyTorch) of decisions, confident ones
  included, and PyTorch int8 lost 10.5 points on MASSIVE English. [Decision
  consistency](../consistency.md) and [Where a request's time goes](../performance.md) repeat both.
- **"Did Laya need the cache for that? No."** True for identical calls. On MPS, a request in a
  batch runs in fp16 and 4 of 1,076 decisions flipped, so on a Mac the cache matters on one machine
  as soon as requests are batched.
- **"A hit almost never writes"** ([cache report](decision-cache-engineering.md), [Decision
  consistency](../consistency.md)). True for a request that repeats more often than every ttl/16.
  Others renew on most hits (0.52 writes per hit in the exactness run), which on SQLite makes a
  hit 2.5x slower.
- **"This makes int8 safe to deploy for repeats"** (cache report). Safe for repeatability: the
  cache fixes int8's first answer. Not for accuracy: that first answer is wrong more often.
- **Language detection "723 → 221 µs".** Holds for English (679 → 214 µs here). Other languages
  gain 1.1x.
- **Open items.** "Real checkpoints and other hardware" is done for Apple silicon, CPU and MPS.
  CUDA, and x86 with real checkpoints (int8 on `fbgemm`), are still open.

## Further benchmarks worth running

- **CUDA.** Laya autocasts CUDA to bf16 or fp16 on every call, so CPU and GPU answers will differ
  more than CPU and MPS fp32 do here; run `bench_backend_consistency.py` and `bench_stages.py
  --device cuda` on a T4 and a newer card. A cache matters most in a mixed CPU and GPU fleet.
- **x86 with real checkpoints.** int8 agreement and accuracy on `fbgemm`, where it is fast, before
  it is recommended anywhere.
- **A real request log.** Replay the retention simulation over hashed production traffic to choose
  `ttl` and `maxsize` from the real repeat pattern rather than a Zipf trace.
- **Redis over a network.** Hit latency, and first-decision-wins between machines whose clocks
  differ: each machine expires decisions by its own clock.
- **A day-long soak.** One SQLite file under mixed traffic for 24 hours: file and WAL growth,
  checkpoint pauses, and the process's memory with the detection memo and question-head cache
  full.
- **A model upgrade.** How many decisions change between two checkpoint revisions: what a new
  fingerprint makes the cache forget, and what a pinned `fingerprint=` would keep replaying.
- **The server with the cache wired in and hits looked up before the inference lock**, under an
  HTTP load generator.

## Reproduce

```bash
git worktree add ../laya-main 4066d5d
python research/scripts/request_sets.py --out requests.jsonl                  # needs `datasets`
python scripts/export_onnx.py --model convaiinnovations/laya --output english.onnx --int8

python research/scripts/bench_branch_ab.py --base ../laya-main --requests requests.jsonl --device mps --rounds 4
python research/scripts/bench_branch_ab.py --base ../laya-main --requests requests.jsonl --device cpu --rounds 4
python research/scripts/bench_branch_same_model.py --base ../laya-main --device cpu
python research/scripts/check_lang_parity.py --base ../laya-main
python research/scripts/bench_backend_consistency.py --requests requests.jsonl \
    --onnx fp32=english.onnx --onnx int8=english.int8.onnx
python research/scripts/bench_equivalent_requests.py --requests requests.jsonl --device mps
python research/scripts/bench_retention.py
python research/scripts/bench_store_scaling.py
python research/scripts/bench_cache_serving.py --requests requests.jsonl --device mps
python research/scripts/bench_stages.py --device mps --variants inference_mode,head_cache_off,mps_fp16
python -c "import runpy, sys, torch; torch.backends.quantized.engine = 'qnnpack'; \
    sys.argv = ['x', '--onnx', 'fp32=english.onnx', '--onnx', 'int8=english.int8.onnx']; \
    runpy.run_path('research/scripts/bench_cpu_fast_path.py', run_name='__main__')"
```

Raw results are in `research/results/*_m4max_20260928.json`.
