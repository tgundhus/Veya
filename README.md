# Veya

Veya is a local AI decision engine for classification, scoring and yes/no questions.
Use it for ticket routing, moderation, guardrails and structured decisions in your own applications.
It runs on your hardware through Python or ONNX Runtime, with HTTP and local MCP interfaces.

Built on [Laya](https://github.com/NandhaKishorM/laya), this fork adds configurable decision
retention, replay across devices and precision modes, and optimizations around inference.
The pretrained models remain upstream Laya checkpoints; the fork maintains its own code,
SDKs, documentation and measurements.

Selected upstream changes are reviewed through Laya 0.4.18, with a
[5 November comparison](https://github.com/tgundhus/Veya/blob/main/docs/reports/decision-engines-2026-11-05.md)
covering newer Laya changes and four related decision engines. Review coverage
does not mean every upstream change has been adopted. The
[runtime adoption report](https://github.com/tgundhus/Veya/blob/main/docs/reports/runtime-adoption-2026-11-05.md)
records the resulting changes, measurements and qualification limits.

The [production capabilities report](https://github.com/tgundhus/Veya/blob/main/docs/reports/production-capabilities-2026-11-05.md)
adds verified model-free replay, cache-aware HTTP scheduling, a shared Python/TypeScript
routing contract, release qualification against reviewed baselines and optional local
JavaScript MCP composition. The canonical English checkpoint fails the illustrative
qualification fixture's quality gate. The
[checkpoint investigation](https://github.com/tgundhus/Veya/blob/main/docs/reports/checkpoint-qualification-repair-2026-11-05.md)
explains why and evaluates an experimental local correction without changing the gates.

[Documentation](https://tgundhus.github.io/Veya/) · [Integration](https://github.com/tgundhus/Veya/blob/main/docs/integration.md) ·
[Benchmarks](https://github.com/tgundhus/Veya/blob/main/BENCHMARKS.md) · [Deployment](https://github.com/tgundhus/Veya/blob/main/docs/production.md)

## Install

```bash
python -m pip install "veya @ git+https://github.com/tgundhus/Veya.git"
# HTTP and MCP support
python -m pip install "veya[serve,mcp] @ git+https://github.com/tgundhus/Veya.git"
```

Python 3.10–3.13. The distribution is `veya`; imports and commands remain `laya`
for compatibility. The HTTP client distributions are `veya-client`, and the
lightweight inference distribution is `veya-onnx`. Native SDK import names remain compatible.
Install it separately from upstream `laya`, since both provide that import.
When upgrading an existing environment, uninstall the former `laya-pro`,
`laya-pro-client` or `laya-pro-onnx` distribution before installing its renamed
replacement. Pip treats these as different projects even though their import
paths overlap; uninstalling the old package afterward can remove the new files.
Source installation is supported until a Veya registry release is published.
For reproducible deployment, append a reviewed commit SHA to the Git URL.

Historical benchmark artifacts retain their original labels, source hashes and
recorded filesystem paths. The local checkout directory can still have its former
name; it is independent of the GitHub repository and package identity.

The first model load downloads a checkpoint. Inference runs locally. For offline use,
download checkpoints beforehand and pass their directories to the Router.
[Installation and platform notes](https://github.com/tgundhus/Veya/blob/main/docs/guide.md#installation).

For an exported ONNX model, the separate `veya-onnx` distribution uses the
`laya_onnx` import and runs without PyTorch or Transformers. Install from a checkout
with `python -m pip install ./runtime/onnx`; it has not been published to a registry.
Optional HTTP and local MCP services are available with
`python -m pip install "./runtime/onnx[serve,mcp]"`. Start them with
`laya-onnx-serve` or `laya-onnx-mcp`; configure local checkpoint and exported graph
pairs as described below. The full package remains the export toolchain and keeps
its current default backend.
[Lightweight ONNX setup](https://tgundhus.github.io/Veya/onnx-runtime/).

## First decision

```python
from laya import DecisionCache, Router, decision_margins

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments and refunds",
                     "technical": "bugs, outages and access problems",
                     "other": "other requests"},
    },
    "urgent": {"type": "noul", "instructions": "Does this need attention today?"},
}

with DecisionCache("decisions.sqlite", ttl=None, maxsize=None,
                   fingerprint="support-policy-v1", required=True) as cache:
    with Router(hooks=[cache]) as router:
        state = "We were billed twice. Please refund the duplicate today."
        result = router.predict(state, questions)
        print(result["answers"]["department"]["choice"])
        print(decision_margins(result))
        router.predict(state, questions)  # an exact repeat replays the stored decision
```

| Question | Returns |
|---|---|
| `choice` | Label, probabilities per option and confidence |
| `score` | Expected level on an ordered rubric and its distribution |
| `noul` | Probability of yes |

Questions in a request share a forward pass. `predict_batch` handles multiple states;
`predict_long` handles documents beyond the context window. Routing selects an English,
multilingual or task-specific checkpoint. Explicit `model` and `lang` overrides are available.

The experimental, opt-in [`ContextCompactor`](https://github.com/tgundhus/Veya/blob/main/docs/compaction.md) uses local decisions
to prune explicitly replayable tool history. Conservative defaults retain uncertain,
protected or incompletely scored content; evaluate retention before using it in an agent loop.

## What the fork adds

- **Decision retention:** fixed or per-decision TTL, optional renewal on use, bounded storage,
  memory or SQLite persistence, and a protocol for your own store.
- **Consistent replay:** the first stored decision wins for exact repeats sharing a store.
  Required mode refuses successful predictions when replay or commit fails.
  Identical requests in one process can share an in-flight computation.
- **Decision margins:** inspect how close an answer is to its decision boundary.
- **Less inference overhead:** reuse question tokens and recent language detections.
- **Deployment controls:** enable the decision cache in HTTP and local MCP, with cache
  statistics and configured retention.
- **Performance modes:** five opt-in budgets for threads, batching and resident models across
  Python, CLI, HTTP and local MCP. Existing defaults remain when no mode is selected.

The cache stores request hashes and result payloads, including option labels and probabilities.
The example keeps decisions without expiry or capacity eviction under an explicit policy
identity; change that identity deliberately when reevaluation is required. A failed storage
operation raises `DecisionCacheError` in required mode. Optional mode remains
fail-open. Preserve the database: required mode cannot detect deletion before startup.
It does not store the original request text. Expiry prevents replay; physical deletion requires
pruning and a policy for database files, journals and backups. A cached answer can still be
wrong. [Cache behavior and retention](https://github.com/tgundhus/Veya/blob/main/docs/consistency.md).

## Integration

| Application | Interface |
|---|---|
| Python, in process | `Router`, `Agent`, `ONNXAgent`, typed schemas and prediction hooks |
| JavaScript / TypeScript, HTTP | [HTTP client](https://github.com/tgundhus/Veya/blob/main/sdk/typescript/README.md) for your own `laya-serve` |
| JavaScript / TypeScript, in process | [ONNX SDK](https://github.com/tgundhus/Veya/blob/main/laya-ts/README.md) |
| Java, in process | [Java SDK](https://github.com/tgundhus/Veya/blob/main/laya-java/README.md) |
| .NET, in process | [.NET SDK](https://github.com/tgundhus/Veya/blob/main/laya-dotnet/README.md) |
| Editor or agent tools | [MCP over stdio](https://github.com/tgundhus/Veya/blob/main/docs/cli-mcp.md), local inference or your own HTTP server |
| Agent frameworks | [LangChain / LangGraph](https://github.com/tgundhus/Veya/blob/main/docs/langchain.md), [LlamaIndex](https://github.com/tgundhus/Veya/blob/main/docs/llamaindex.md), [CrewAI](https://github.com/tgundhus/Veya/blob/main/docs/crewai.md) |

The SDKs have different capabilities. Veya's persistent decision cache runs in the Python engine;
HTTP clients and MCP can use that engine's cache. Native SDKs do not implement the same
persistence policy. [Capability table and examples](https://github.com/tgundhus/Veya/blob/main/docs/integration.md).

Start a local HTTP server with `laya-serve`, then call `POST /v1/systemone` or
`POST /v1/systemone/batch`. Set these variables in the server's environment for a
persistent decision policy without expiry or capacity eviction:

```text
LAYA_CACHE_PATH=decisions.sqlite
LAYA_CACHE_TTL=none
LAYA_CACHE_MAXSIZE=none
LAYA_CACHE_FINGERPRINT=support-policy-v1
LAYA_CACHE_REQUIRED=1
```

The same cache variables apply to `laya-mcp-server`. Authentication, request bounds,
concurrency and offline setup are covered in the [deployment guide](https://github.com/tgundhus/Veya/blob/main/docs/production.md).

For retained answers without loading models after restart, prepare a
[replay manifest and pin the SQLite store identity](https://github.com/tgundhus/Veya/blob/main/docs/replay.md).
Approved HTTP hits can then use the [separate cache worker](https://github.com/tgundhus/Veya/blob/main/docs/cache-fast-path.md).
To compose several operations in one local MCP call, install the `codemode` extra
and start `laya-mcp-server --mode code`; see the
[Code Mode guide](https://github.com/tgundhus/Veya/blob/main/docs/mcp-code-mode.md) for its limits.
Before changing a model, precision or batching policy, run
[release qualification](https://github.com/tgundhus/Veya/blob/main/docs/release-qualification.md)
on reviewed deployment cases. The
[routing contract](https://github.com/tgundhus/Veya/blob/main/docs/routing-parity.md)
documents supported Python/TypeScript parity and migration costs.

Use `Router(performance_mode=2)`, `laya --performance-mode 2`, or
`LAYA_PERFORMANCE_MODE=2` in the engine's environment to choose a resource budget.
Explicit settings override profile values; `performance_mode="default"` ignores the environment
for an instance. Modes keep the same model and precision. Higher modes are not necessarily
faster on your hardware. [Mode budgets and measurements](https://github.com/tgundhus/Veya/blob/main/docs/performance-modes.md).

## Measurements

On a pinned real English CPU checkpoint, bounded loading reduced peak working set by
764 MiB (28%), with unchanged steady memory and inference time. The standalone ONNX
runtime reduced startup to the first answer from 5.37 to 1.78 s on the same graph;
warm inference stayed the same and all 12 typed answers matched exactly.
[Loading measurements and limits](https://github.com/tgundhus/Veya/blob/main/BENCHMARKS.md#loading-memory-and-standalone-onnx-31-october-2026).

The October CPU check used a pinned English checkpoint, fp32 and four threads. Median single
inference was 571.5 ms; a SQLite repeat was 0.148 ms. All 12 answers agreed between serial and
grouped batch execution. Workstation load was uncontrolled and this small check does not
measure accuracy. [Current results and methods](https://github.com/tgundhus/Veya/blob/main/docs/reports/production-review.md).

These archived results used published checkpoints on an Apple M4 Max in September 2026,
against Laya 0.3.20/0.3.21. They describe those runs, not a guarantee for newer versions or
another machine. [Method and raw results](https://github.com/tgundhus/Veya/blob/main/docs/reports/real-checkpoints-m4-max.md).

| Workload | Upstream run | Veya run |
|---|---|---|
| Repeated three-question ticket, CPU | 143.9 ms | 0.037 ms, memory cache |
| Repeated three-question ticket, MPS | 34.1 ms | 0.058 ms, memory cache |
| Stream with 70% repeats, MPS | 16.8 requests/s | 61.0 requests/s |
| 16 identical new requests, CPU | 16 forward passes, 1.26 s | 1 forward pass, 0.22 s |
| Shared-model new request, CPU | 152.7 ms | 151.8 ms |
| Cached decisions replayed after SQLite restart | — | 20,000 of 20,000 |

New requests are dominated by inference; cache speed depends on repeat frequency.
Earlier parity checks covered 841 requests in 11 languages. Precision changes can change
decisions, so validate fp16, bf16 or int8 on your own data before adopting them.
The [October review](https://github.com/tgundhus/Veya/blob/main/docs/reports/production-review.md) records current checks, fixes,
measurements and verification limits separately from these archived results.

## Limits and version policy

This version incorporates upstream **0.3.29** and selected correctness and integration changes
reviewed through **0.4.18**. The distribution remains 0.3.29 and retains Veya's English
fallback for ambiguous Latin text. This is selected upstream coverage, not complete behavioral
parity; the [30 October review](https://github.com/tgundhus/Veya/blob/main/docs/reports/upstream-review-2026-10-30.md)
records adopted changes, measurements and limits.

Validate accuracy and confidence on representative labelled data. Large choice sets,
underspecified criteria and ordinal scores remain difficult. Replay improves repeatability,
not accuracy; an expired, evicted or unavailable entry is computed again.
See [measured limits](https://github.com/tgundhus/Veya/blob/main/docs/limits.md) and the [evaluation harness](https://github.com/tgundhus/Veya/blob/main/docs/evals.md).

## Development and license

[Contributing](https://github.com/tgundhus/Veya/blob/main/CONTRIBUTING.md) covers test suites, required lint and compile gates, and strict
documentation builds. Changes should include a runnable regression check and measured evidence
when they alter performance or numerical behavior.

Laya was created by Nandakishor M and developed by Convai Innovations. Veya's additions
are maintained by Tobias Gundhus. The engine, checkpoints and fork are licensed under
[Apache 2.0](https://github.com/tgundhus/Veya/blob/main/LICENSE). Veya is an independent fork, with no endorsement implied.
