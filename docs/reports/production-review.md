---
title: Veya review, October 2026
description: Upstream reconciliation, cache and integration fixes, SDK validation and reproducible measurements for Veya.
---

# Veya review, October 2026

The review started on 8 October from `692cea9`, which still used upstream 0.3.21. The
pending branch `merge/upstream-0.3.29` ended at `e29aa6d`, with merge `2d0be4e` incorporating
upstream release `e08843b`. That branch is incorporated here. Selected correctness fixes and
the Java and .NET SDK sources are taken from upstream 0.4.1, `1adc59f`.

Veya retains the English default for ambiguous Latin text. Upstream 0.4 switched it to
multilingual. Choice criteria retain insertion order: sorting them would change model inputs.
This fork's version remains 0.3.29 and does not imply complete upstream 0.4.1 parity.

## Review scope

The review covers inference and routing, batches and long documents, confidence policies,
checkpoint identity, tokenizer reuse, cache replay and retention, HTTP and MCP lifecycles,
SDK request/response contracts, package metadata, CI registration and documentation builds.
Existing hardware-specific results are retained as historical measurements.

## Changes

| Area | Verified issue and resulting behavior |
|---|---|
| Cached payloads | Malformed JSON structures or token totals could reach prediction callers. Invalid entries now fall back to computation. |
| Bounded memory cache | Capacity eviction could remove a live entry while expired entries remained. Reclaim expired entries before live eviction, with an expiry guard to avoid scanning a full live cache on every insertion. |
| Cache identity | Different subfolders or replaced local artifacts could share a fingerprint. Subfolders and local weight, encoder configuration and tokenizer identity now contribute to automatic identity. |
| Cache serialization | Numeric schema keys could become strings and collide with string keys. Typed encoding preserves supported key types. |
| Tokenizer reuse | Replacing tokenizer artifacts could leave a stale parsed tokenizer; parser retention was unbounded. Track artifact metadata and bound the parser cache. |
| ONNX | A single-option question could feed a TopK graph with fewer than two marker columns. Padding handles that shape. |
| Server lifecycle | An app's first shutdown left later lifespans with a closed executor. Each lifespan now owns its worker and can retry initialization after configuration failure. |
| HTTP and MCP cache | The process builders did not install the Pro decision cache. Shared opt-in configuration exposes fixed or sliding retention, memory/SQLite stores and statistics, with Docker Compose and Nix controls. |
| Remote MCP | Batch confidence controls could be dropped. Forward them consistently. |
| TypeScript HTTP | Custom checkpoints, null state handling and batch requests had contract gaps. Validate and support the server's actual request/response shapes. |
| Framework adapters | Invalid CrewAI or LlamaIndex labels could silently select the wrong tool. Validate labels and fallback indices. |
| Training and structured calls | Port upstream device and update-budget handling, UTF-8 evaluation output, explicit batch model selection and the final Kaggle convergence helper. Reject dry runs without usable training rows. |
| Native SDKs | Restore Java and .NET sources and add CI builds. Java rejects reentrant async hooks and drains eviction callbacks even when one fails. |
| Batch schema grouping | Numeric and string question IDs or choice labels could share a group and receive the wrong keys. Typed signatures keep those requests distinct while grouping identical schemas. |
| Distribution | Metadata and install links named upstream or an old fork URL. Use the `veya` distribution and this repository, retaining Python import compatibility. |
| Documentation | Canonical URLs, navigation, sitemap and crawl directives now describe Veya's own site. |

The cache key layout changes to version 2. Existing entries are not replayed under the new
layout; they remain on disk until expired entries are pruned or the store is cleared.
This intentionally causes fresh inference after the upgrade.

Unversioned local checkpoints use verified digests when supplied, otherwise file metadata
captured at load. This detects ordinary artifact replacement, not edits preserving the same
metadata. Use explicit, versioned fingerprints when sharing identity across machines.

The new upstream TypeScript hook timeout was deferred: timed-out hooks could still mutate
returned results. Native TypeScript retains its existing awaited-hook behavior.

## Verification and measurements

The host is Windows on an Intel i9-13900KF, with an RTX 4080 present. The review environment
uses Python 3.12, CPU PyTorch, ONNX Runtime, HTTP/MCP extras and the documentation dependencies.
This does not establish CUDA, Apple MPS or TileLang behavior.

The shared Python suite list, required lint and compile gates, SDK tests, packed-package
consumers and the strict documentation build were run. The built site contains 47 sitemap
URLs under `tgundhus.github.io/Veya/`; a runnable check verifies descriptions, canonical URLs and crawl
directives on its entry pages.

| Check | Result |
|---|---|
| Shared Python suite list | All 91 suites passed; the pytest component reported 420 passed and 1 skipped |
| Core cache and API contracts | 247 consistency checks and 626 API contract checks passed |
| HTTP, cache and remote MCP | 328 tests passed |
| Java, JDK 17 and 21 | 966 passed and 16 skipped on each JDK |
| .NET, Release build | No warnings or errors; 755 passed, 682 skipped |
| Native TypeScript | 655 tests, build, type consumer and packed-package checks |
| TypeScript HTTP | 31 tests, packed ESM/CJS consumers, and live single/batch transport parity |
| Python ONNX | 3 actual English export and inference tests, including single-option parity |
| Python distribution | Wheel and source distribution built and passed `twine check` |

Java and .NET source-suite skips require the full pretrained tokenizer or exported model
fixtures, which were not configured for those runs.
Synthetic native fixtures establish API behavior, not pretrained decision quality. Real Python
ONNX checks used English Hub snapshot `7b928d828b7b0e022f929d9bd2e44165aa270148`; the latency
measurement below used a separate pinned snapshot.

A separate real native SDK smoke used that pinned `55cf4c4...` English snapshot and a split
ONNX export. TypeScript and Java singles and batches, plus .NET singles, matched Python
for two states asking choice, score and yes/no questions. Publicly rounded numeric values
had maximum difference zero; available usage fields matched exactly. The exporter also
checked three shapes within `1e-4`, with maximum logit difference `6e-7`. This small check
does not establish multilingual accuracy or every native method.
[Native checkpoint inputs and results](https://github.com/tgundhus/Veya/blob/main/research/results/native_sdk_smoke_windows_i9_20261009.json).

Docker and Nix deployment files passed static contract checks; this Windows environment did
not have Docker or Nix for runtime builds. Hardware-specific fast paths and the full labelled
accuracy sweep remain outside the measurements performed here.

Three CodeQL alerts were assessed statically as fixture false positives: fixed fictional billing
text in two tests and an intentional `0640` permission-preservation test. Query-specific comments
explain those sinks; production paths remain scanned. This triage is not a separate exhaustive
security audit.

### English checkpoint on CPU

The checkpoint is `convaiinnovations/laya` at revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, fp32, four CPU threads. Four states each ask
three typed questions. Latency samples use three repeats after warmup; replay uses 15 repeats.

| Workload | Median |
|---|---|
| One state, three questions | 571.46 ms |
| Four states, serial | 3,300.48 ms |
| Four states, batch size 2 with length grouping | 2,598.10 ms |
| SQLite replay of the stored answer | 0.1482 ms |
| Long input, 1,081 state tokens in 14 windows | 12,035.40 ms |

Repeated payloads were identical. Serial and grouped batch answers had zero decision flips
and a maximum probability delta of zero across 12 answers. Long prediction reported no dropped
state tokens or secondary truncation. Replay avoided 17 model forward passes and worked after
closing and reopening SQLite. Local fingerprint collection took a median 0.0799 ms at load.

Workstation load was uncontrolled: single-call samples ranged from 396.58 to 683.73 ms and an
earlier exploratory run was faster. These figures establish the measured execution paths;
they are too small and variable to support general latency or accuracy claims.
[Raw checkpoint results](https://github.com/tgundhus/Veya/blob/main/research/results/review-2026-10-08-cpu.json).

### Cache capacity and retention

The isolated memory-store comparison alternates the pending merge baseline `e29aa6d` with
the reviewed implementation, five repeats, 100,000-entry capacity:

| Workload | Before | After |
|---|---|---|
| 50,000 insertions into a full live store, median per insertion | 1.323 us | 0.880 us |
| 5,000 insertions with 1,000 expired entries, median per insertion | 0.760 us | 4.053 us |
| Original live entries retained after those mixed-expiry insertions | 94,000 | 95,000 |

Reclaiming expired entries preserves 1,000 additional live decisions at the cost of a scan.
An expiry guard avoids that scan when all entries are live. This is a byte-store benchmark,
not a model or end-to-end service measurement; microsecond timings vary with host activity.
[Raw capacity comparison](https://github.com/tgundhus/Veya/blob/main/research/results/cache_capacity_windows_i9_20261008.json).

The broader model-free cache benchmark records increased request overhead in the final run:

| Router request, instant fake model | Before | After |
|---|---|---|
| Uncached control | 42.627 us | 52.094 us |
| Memory cache hit | 87.686 us | 107.832 us |
| SQLite cache hit | 69.929 us | 121.185 us |

Even uncached control timings moved substantially. Stronger payload validation has costs;
these sequential runs cannot isolate its causal latency effect. There is no claim that every
cache operation became faster.
[Raw cache comparison](https://github.com/tgundhus/Veya/blob/main/research/results/cache_review_windows_i9_20261008.json).

The retention harness exercised 320,000 requests with fixed, sliding and callable TTL policies
across memory and SQLite stores, with zero outcome disagreements. Capacity workloads added
28,004 writes. The memory store remained at its 1,000-entry limit; SQLite reached 1,099 rows
between periodic sweeps, confirming that its bound is a target rather than a strict ceiling.

A fresh process replayed all 2,000 stored decisions without computation. A changed fingerprint
and a simulated 31-day expiry each caused 2,000 fresh computations. Rows from the old model
remained until maintenance; expiry prevents reuse without promising physical erasure.
[Raw retention and restart results](https://github.com/tgundhus/Veya/blob/main/research/results/retention_review_windows_i9_20261008.json).

### Reproducing the checks

Use an environment installed from this checkout with the relevant extras. The checkpoint
benchmark accepts a pre-downloaded local snapshot, keeping inference offline.

```bash
python scripts/test_suites.py
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
python -m pytest tests/test_onnx.py -q
python research/scripts/bench_review.py --checkpoint /path/to/snapshot --out review.json
python research/scripts/bench_cache_capacity.py --baseline e29aa6d --out capacity.json
python research/scripts/bench_decision_cache.py --entries 10000 --out cache.json
python research/scripts/bench_retention.py --sections exactness,restart --restart-decisions 2000 --out retention.json
zensical build --strict --clean
python scripts/check_site.py
```

For the cache before/after comparison, run `bench_decision_cache.py` in separate checkouts of
`e29aa6d` and this change using the same Python environment. Follow each SDK's README for
native build commands. CI includes Java 17/21 and .NET Windows/Linux lanes; the local native
SDK checks here ran on Windows.

Published pretrained accuracy tables are not recomputed by synthetic SDK tests. Use the
[evaluation harness](../evals.md) with your own labelled workload before deployment.

## Publishing the documentation

The documentation workflow is scoped to `tgundhus/Veya`, with canonical site URL
`https://tgundhus.github.io/Veya/`. GitHub Pages now uses **GitHub Actions** to publish
the generated documentation. The earlier `laya.xgnd.me` address had no public DNS records;
the working GitHub Pages address is used instead, with HTTPS enforced.

To move to a custom domain later, configure it in Pages settings and DNS, then update the
site URL, documentation links, robots sitemap and `scripts/check_site.py` together. A root
CNAME file alone does not configure an Actions-based Pages deployment. See [GitHub's publishing
source guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
Verify the site in Google Search Console and submit `/veya/sitemap.xml`. The project
`robots.txt` is published for reference; crawler rules apply only to the host's root
`/robots.txt`. [Google's robots.txt specification](https://developers.google.com/crawling/docs/robots-txt/robots-txt-spec).
Canonical URLs and sitemap entries communicate the preferred pages; they do not guarantee indexing or
ranking. [Google's canonical URL guidance](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls).
