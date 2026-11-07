---
title: Veya naming migration and CPU refresh
description: Primary Veya APIs, compatibility aliases, fresh CPU timings and checkpoint parity.
---

# Veya naming migration and CPU refresh

Veya is now the primary name for the Python package, commands, environment variables,
MCP tools, HTTP clients and native SDKs. The implementation lives in `veya/`;
`veya-ts/`, `veya-java/` and `veya-dotnet/` hold the local native runtimes.
The distribution version remains 0.3.29. Checkpoint weights and inference defaults
are unchanged.

## Migration

| Interface | Primary name | Compatibility |
|---|---|---|
| Python inference | `veya`, `veya.router`, `veya.hooks` | `laya` imports redirect to the same modules and classes |
| Python HTTP | `veya_client.VeyaClient`, `VeyaAPIError` | `laya_client.LayaClient` and legacy error names remain aliases |
| Lightweight ONNX | `veya_onnx`, `veya-onnx-serve`, `veya-onnx-mcp` | Legacy `laya_onnx` imports and commands remain aliases |
| Commands | `veya`, `veya-serve`, `veya-mcp-server`, `veya-evals`, `veya-train`, `veya-replay` | The corresponding `laya` commands still work |
| Configuration | `VEYA_*` | Python reads `LAYA_*` when the primary variable is absent; explicit empty values mask legacy settings |
| MCP | `veya_predict`, `veya_route` and the other `veya_*` tools | The ten classic `laya_*` names remain registered aliases |
| TypeScript HTTP | `Veya`, `VeyaOptions`, `VeyaAPIError` from `veya-client` | Legacy `Laya` exports share the same constructors |
| TypeScript local | `veya-ts` | New package references are required; supported Node environment settings retain legacy fallbacks |
| Java | `com.convaiinnovations.veya`, `veya-java` | Update package imports and artifact references |
| .NET | `Veya.Onnx`, namespace `Veya` | Update package and namespace references |

Integration classes use `VeyaRouter`, `VeyaGuardrail` and other `Veya*` spellings.
Their previous Python `Laya*` names remain aliases. Imports share hook registries,
cache state and class identities; importing both names does not load a second engine.
Legacy imports retain canonical module metadata, including for relative imports.

Upstream `convaiinnovations/laya` checkpoint IDs and upstream credits still name
Laya. Historical result labels, hashes and recorded paths remain unchanged.
SQLite retains the existing `laya_decisions` and `laya_cache_metadata` table names;
renaming a product must not create an empty replacement decision store.

Uninstall former `laya-pro`, `laya-pro-client` and `laya-pro-onnx` distributions
before installing their replacements. Their legacy import files overlap. Use a
separate environment for upstream `laya`. The new setup writes the English bundle
to `models/veya`, matching the verification manifest. Existing checkpoint directories
remain valid when supplied explicitly; this migration does not move local weights.

## CPU measurements

Windows 11, Intel Core i9-13900KF, 32 logical CPUs, Python 3.12.14 and
PyTorch 2.14.1+cpu. The real English checkpoint has 421.3M parameters and revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`. Both runs used eager fp32 and two Torch
threads, with three warm-ups followed by 15 uninstrumented calls per workload.
Separate instrumented calls identify the inference stages. Cache timings use 50
memory-cache hits after priming. Loading and HTTP overhead are excluded.

| Workload | Before, median ms | Veya, median ms | Change |
|---|---:|---:|---:|
| Ticket, one question | 179.146 | 171.993 | -3.99% |
| Ticket, three questions | 511.658 | 509.616 | -0.40% |
| Ticket, ten questions | 1615.797 | 1609.739 | -0.37% |
| Email, three questions | 2435.051 | 2480.429 | +1.86% |
| Ticket, three questions, cache hit | 0.058 | 0.056 | -3.45% |
| Email, three questions, cache hit | 0.059 | 0.059 | 0.00% |

These are sequential workstation runs with uncontrolled load, not evidence that
renaming improves performance. The baseline harness recorded summary statistics;
the refreshed harness additionally records individual timing samples, source
digests, timestamp and checkpoint revision. Local checkpoint paths no longer
automatically label real weights as synthetic; synthetic runs must explicitly
set `--synthetic-weights`.

A separate offline comparison extracts pre-rename commit
`a5477bdc4d63647c5d5bd740ab989c6ffa11bfe6` and runs it in another process against
the same requests and cached checkpoint. All 17 typed answers, checkpoint revision
and token counts match exactly. This is a small migration parity check, not a
task-accuracy evaluation. GPU and multilingual benchmarks were not rerun here.

- [Before measurement](../../research/results/veya_rename_before_cpu_20261107.json)
- [Refreshed samples and stage timings](../../research/results/veya_rename_after_cpu_20261107.json)
- [Raw before/after answers](../../research/results/veya_rename_parity_cpu_20261107.json)
- [Summary and artifact digests](../../research/results/veya_refresh_cpu_20261107.json)

The [6 November ordinal continuation](ordinal-continuation-2026-11-06.md) remains
experimental. Its fresh holdout has 20/36 scores within ±0.25, 10/15 correct choices
and 9/10 correct binary answers. Score and choice fail their 80% gates. Those results
describe a local adapter, not a replacement default checkpoint.

## Verification

All 122 shared suite files have been validated, including a successful rerun of the
packaging link check. The pytest group reports 1,641 passed, one optional dependency
skip and 4,238 passing subtests. Ruff, compilation and the strict documentation build
pass.

The API contract covers primary and legacy module identity, live environment
fallback, explicit-empty precedence and canonical import metadata. It passes 928
checks. The Python HTTP client passes 69 installed-package tests with one platform
skip, both from its wheel and from a rebuilt sdist. The lightweight ONNX package
passes isolated inference, HTTP, MCP, code mode and model-free replay consumers,
including an outside-checkout source rebuild. Its
[package report](../../research/results/veya_onnx_packaging_20261107.json) records
the bundled source digests. The full wheel also resolves both Python namespaces
outside the checkout.

The TypeScript runtime passes 1,339 tests plus its external packed-package test;
the HTTP client passes 34 tests including ESM, CommonJS and package installation.
The .NET solution builds without warnings and its runnable tests pass; graph-backed
cases skip when exported checkpoints are absent (755 passed, 682 skipped). Java
compiles with JDK 17, `-Xlint:all -Werror`, and passes 1,179 JUnit cases; one skips
and 13 abort by assumption without checkpoint artifacts. Gradle cannot establish its
daemon loopback connection on this host, so compilation and JUnit launch directly
using the cached dependency JARs. Java fixtures are regenerated
against the current Python contract. Schema signatures are compared with normalized
JSON separators while preserving fields, option order and whitespace inside text.

## Reproduce

From the repository root in PowerShell, using the same cached checkpoint:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
python research/scripts/bench_stages.py --device cpu --models english --repeats 15 --warmup 3 `
  --scenarios "english ticket, 1 question;english ticket, 3 questions;english ticket, 10 questions;english email, 3 questions" `
  --variants "" --out current_cpu.json
python research/scripts/check_rename_parity.py --baseline a5477bdc --out parity.json
```

The parity command exits nonzero if any answer, token count or checkpoint revision
changes. The generated Colab benchmark now installs and imports Veya; edit
`research/scripts/build_benchmark_nb.py` and regenerate
`research/scripts/veya_benchmark_colab.ipynb` when changing that harness.
