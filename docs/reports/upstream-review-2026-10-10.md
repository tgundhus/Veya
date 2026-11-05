---
title: Selected upstream review, 10 October 2026
description: Selected Laya 0.4.2 fixes, local evaluation tools, SDK checks and measured integration costs.
---

# Selected upstream review, 10 October 2026

This review covers the 39 upstream commits from
[`1adc59f7e371deb601fcfa18a14e25db238addcc`](https://github.com/NandhaKishorM/laya/commit/1adc59f7e371deb601fcfa18a14e25db238addcc)
to [`68804629e8ccd9d616d48a40e87de9fabbeae069`](https://github.com/NandhaKishorM/laya/commit/68804629e8ccd9d616d48a40e87de9fabbeae069),
the 0.4.2 release. The [source comparison](https://github.com/NandhaKishorM/laya/compare/1adc59f7e371deb601fcfa18a14e25db238addcc...68804629e8ccd9d616d48a40e87de9fabbeae069)
records the review boundary, rather than a claim that every upstream change was adopted.

Veya remains version 0.3.29. Its English fallback, criteria insertion order and typed cache
signatures are preserved. This change does not replace checkpoints, fit temperatures or
recompute pretrained accuracy and calibration tables. The [earlier review](production-review.md)
records the preceding merge and cache work.

## Adopted changes

| Area | Result and upstream source |
|---|---|
| Python shortlist and tournament | Remove the original `option_order` after narrowing, with one warning per question per call. The previous permutation failed Agent validation. Caller data and passthrough questions stay intact. [Source](https://github.com/NandhaKishorM/laya/commit/f707d29) |
| Router diagnostics | Warn when a custom name is one edit from a built-in, including an adjacent letter swap. The custom name remains separate. Warnings promoted to errors leave registration and attachment unchanged. [Source](https://github.com/NandhaKishorM/laya/commit/d728d6f) |
| HTTP responses | Add `x_jev_confidence` to choice and score answers for consumers using Jev's confidence definition. Existing confidence fields retain their meaning; strict Jev projection removes extensions. [Source](https://github.com/NandhaKishorM/laya/commit/c932543) |
| Compose | Forward extra checkpoint registrations, idle unloading, batch token limits and strict Jev controls to the server. [Registration controls](https://github.com/NandhaKishorM/laya/commit/5067f79), [batch and wire controls](https://github.com/NandhaKishorM/laya/commit/b0aed97); [configuration guide](../docker.md#server-configuration) |
| Local evaluation | Add saved-run agreement comparison, local student evaluation, paired case bootstrap intervals and a CLI. Validate aligned inputs and preserve unknown teacher revisions. [Source](https://github.com/NandhaKishorM/laya/commit/5aae58c), [usage and limits](../evals.md#comparing-a-student-with-repeated-teacher-decisions) |
| Java | Port single-option graph padding, grouped Router batches, tournaments and hook/error-ordering fixes. Further review fixed hook-rewritten batch cardinality and propagation of outer async deadlines. [Single-option fix](https://github.com/NandhaKishorM/laya/commit/1e68e9a), [batching](https://github.com/NandhaKishorM/laya/commit/d76eacb), [tournament](https://github.com/NandhaKishorM/laya/commit/71e43b7), [hook ordering](https://github.com/NandhaKishorM/laya/commit/7a79168) |

## Measured costs

The Windows/Python 3.12 setup benchmark alternates baseline `aba90f3` and this implementation,
nine repeats of 10,000 calls. Warnings are ignored during timing; no checkpoint or model
forward runs.

| Operation | Before, median µs | After, median µs |
|---|---:|---:|
| Router registration of built-in `english` | 4.664 | 4.593 |
| Router registration of custom `papers` | 4.909 | 12.640 |
| Router registration of typo `englsh` | 4.905 | 14.336 |
| Shortlist without `option_order` | 24.596 | 24.398 |

The diagnostic adds configuration cost, depending on the name; it runs outside prediction.
Stale-order timings were noisy and the baseline input failed real Agent validation, so no
speed improvement is claimed. [Harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_router_shortlist_setup.py),
[raw samples](https://github.com/tgundhus/Veya/blob/main/research/results/router_shortlist_setup_windows_20261010.json).

HTTP response enrichment took medians of **6.478 µs for three answers** and **154.068 µs for
64 answers**, seven repeats of 2,000 calls. These measure response enrichment alone, excluding
inference, serialization and networking. Workstation load was uncontrolled in both benchmarks.
[Harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_http_extensions.py),
[raw samples](https://github.com/tgundhus/Veya/blob/main/research/results/http_extensions_windows_20261010.json).

The Java English smoke timed the same two requests through the Router: **1,501.15 ms
sequentially** versus **1,298.06 ms in one batch**, median whole-call time with two ONNX Runtime
threads. Each mode had two warmups and three timed repeats, alternating execution order;
answers and usage stayed identical. These two sample inputs on an active workstation do not
establish a general throughput improvement.
[Samples and settings](https://github.com/tgundhus/Veya/blob/main/research/results/java_upstream_parity_windows_20261010.json).

## Checks and limits

All 92 suites in the shared local Python runner passed; its pytest component reported
**497 passed and one skipped**. Targeted checks passed: 227 shortlist assertions, 802 routing
assertions, 36 Router batch tests, 219 evaluation/agreement/shortlist-evaluation tests, 331
HTTP/server tests and 32 TypeScript HTTP tests. Contract checks passed: 50 evaluation API
assertions and 674 hook/API assertions.
A live TypeScript HTTP client also passed single/batch transport checks against local Python
inference in offline mode, including the added confidence fields.

Java JDK 17 and 21 each passed **1,177 tests**, with **23 skipped** and no failures; the JDK 17
build also passed compilation with warnings as errors, Javadoc and packaging. Three added
regressions cover extra or empty hook-rewritten batch states and an outer async-hook deadline.

Reference regeneration pins `tokenizers==0.23.2`: that version reproduced the committed Java
pre-tokenizer table exactly with `gen_pretokenizer_tables.py --check`. An unpinned tokenizer
can change Unicode classification and version-stamped output, producing fixture drift even
when the runtime code is unchanged. [Pinned toolchain](https://github.com/tgundhus/Veya/blob/main/laya-dotnet/tools/requirements-regen.txt),
[table generator](https://github.com/tgundhus/Veya/blob/main/laya-java/scripts/gen_pretokenizer_tables.py).

The retained English checkpoint `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` matched Python at
returned precision in a small Java check: two states with choice, score and yes/no questions,
singles and batches, single-option choice/score, Router calls and a 20-label tournament.
Maximum reported difference was zero at tolerance `1e-4`. Multilingual and typed-checkpoint
graph checks remain skipped. This small workload does not establish general accuracy or calibration.
[Raw parity results](https://github.com/tgundhus/Veya/blob/main/research/results/java_upstream_parity_windows_20261010.json),
[Python comparison harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/java_upstream_smoke.py),
[Java harness](https://github.com/tgundhus/Veya/blob/main/research/scripts/JavaUpstreamSmoke.java),
[regression suites](https://github.com/tgundhus/Veya/tree/main/laya-java/laya-java/src/test/java/com/convaiinnovations/laya).

The [agreement example](https://github.com/tgundhus/Veya/blob/main/examples/evals/teacher_agreement.py)
reproduced the saved report through the CLI on 32 fixed cases from a pinned public dataset.
Only **1 of 32 student decisions** matched the teacher reference, while the teacher repeat
matched **26 of 32**. This is poor agreement on a small academic panel, not production
accuracy. The teacher's immutable model revision is unknown, and its prompt differs from
Laya's typed inputs. No teacher API is required: observations are saved and student inference
runs locally.
[Pinned metadata and measured summary](https://github.com/tgundhus/Veya/blob/main/research/results/agreement_smoke_windows_20261010.json).

TypeScript timeout changes remain deferred: the late compilation fixes do not complete timeout
validation or stop an abandoned hook from modifying its context. Native TypeScript retains
awaited hooks. The .NET multilingual default is not adopted; Veya keeps its English
fallback. Docker was unavailable on the local Windows review host, so Compose environment
controls were checked statically here. GitHub CI on [PR 9](https://github.com/tgundhus/Veya/pull/9)
and [PR 10](https://github.com/tgundhus/Veya/pull/10) passed 28 checks per PR, including
Docker HTTP and Linux amd64/arm64 smoke tests. Those runs precede the final Java follow-up.
