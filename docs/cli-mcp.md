# Command line and MCP server

Veya has two local interfaces for trying the same structured-decision engine:

| Interface | Use it for | Transport |
|---|---|---|
| `veya` | quick checks and interactive exploration from a terminal | command line |
| `veya-mcp-server` | connecting an MCP client or agent to Veya's built-in tools | MCP over stdio |

Choose the CLI when you are the person reading the result. Choose MCP when another process needs
a stable tool interface. Both use Veya's `Router` to select a checkpoint and return typed
`choice`, `score`, and `noul` decisions; neither is an open-ended question-answering or text
generation interface.

For the routing decision and typed-question examples, see [Routing](guide.md#routing) in the guide.
For confidence and built-in workflows, see [confidence gating](guide.md#confidence-and-abstention) and [workflow presets](guide.md#presets) in the guide.

## 1. Command line

Installing the package installs the `veya` entry point. Run `veya --help` for the complete
option list.

```bash
python -m pip install "veya @ git+https://github.com/tgundhus/Veya.git"
veya --help
```

### Evaluation CLI

The package also installs `veya-evals`. The main CLI exposes the same evaluation commands
through `veya eval`:

```bash
veya eval --help
```

See the [Evaluation harness](evals.md) guide for datasets, metrics, and baseline gates.

### Training CLI

`veya-train`, also reachable as `veya train`, fine-tunes a checkpoint from a CSV or JSONL of labelled
decisions. See [Fine-tune with `veya-train`](finetune.md#fine-tune-with-veya-train).

```bash
veya-train --help
```

### Route without loading a checkpoint

With text and no prediction flag, the CLI calls `Router.route`:

```bash
veya "I was charged twice, please refund it"
```

The output names the selected checkpoint, explains why it was selected, and shows detected
language information when available. Routing alone does not download or build a checkpoint, so
it is a quick offline check of the routing decision.

Use `--json` when another local script should consume the decision:

```bash
veya "I was charged twice, please refund it" --json
```

### Run a prediction

`--predict` runs the full typed prediction and loads the routed checkpoint on first use. The
first load needs access to the Hugging Face Hub; later runs use the local cache.

```bash
veya "Classify this support request" --predict
veya "Classify this support request" --predict --json
```

`--json` prints the complete result as JSON. Without it, the CLI prints each answer together
with its choice probability, score, or `noul` value, plus the routing decision.

The main controls are:

- `--model english|multilingual|typed-decisions` pins a checkpoint instead of auto-routing.
- `--lang en|de|...` supplies an explicit language code instead of automatic detection.
- `--lang-guess en|de|...` supplies a soft hint that routing reads after `--lang` and before its
  own detector; a hint that resolves to nothing falls through, so it nudges the checkpoint without
  forcing it.
- `--task NAME` forces the typed-decisions workflow instead of detecting it.
- `--device cpu|cuda|...` passes a device choice to the Router.
- `--performance-mode 1|2|3|4|5|default` chooses a resource profile; `default` ignores
  `VEYA_PERFORMANCE_MODE`. Without either setting, existing defaults remain.
- `--json` emits machine-readable output.
- `--explain` adds language-detection diagnostics to routing output, including the
  analysis stage and the source path when a nested field changes the result.

### Use a built-in preset

A preset supplies a ready-made question set and implies prediction, so `--predict` is not needed:

```bash
veya "My payment failed twice" --preset triage
veya "Ignore all previous instructions" --preset guard --json
```

The CLI presets are `email`, `guard`, `moderation`, `router`, and `triage`. The CLI places the
text under the state field expected by the selected preset; `--predict` uses the router
question set's `request` field. Presets are useful for a quick local check, but their questions
are still domain decisions: inspect the preset and validate it on your own data before using it
as an application policy.

### Explore interactively

With no text argument, the CLI opens a small prompt:

```bash
veya
# veya> Classify this request
# veya> quit
```

Press Enter to run each request. An empty line, `quit`, `exit`, or `Ctrl-D` ends the session. The
interactive loop reuses one Router, so it is a convenient way to compare several inputs without
writing a script.

### Failures are visible

The CLI handles invalid values and common dependency, download, and runtime failures at the
application boundary. It prints a diagnostic to stderr and returns exit code `2` instead of
showing an unhandled traceback. If a first-use checkpoint download fails, check dependency
installation, Hub access, and the selected device before retrying.

## 2. Built-in MCP stdio server

The MCP server is an optional extra. The core package does not install the `mcp` dependency:

```bash
python -m pip install "veya[mcp] @ git+https://github.com/tgundhus/Veya.git"
veya-mcp-server
# equivalent module form:
python -m veya.mcp.server
```

For an installation without Torch or Transformers, use
`python -m pip install "./runtime/onnx[mcp]"` and launch `veya-onnx-mcp`, or
`python -m veya_onnx.mcp.server`. Configure its local checkpoint and exported graph
pairs using [Lightweight ONNX runtime](onnx-runtime.md#http-and-local-mcp).
The full package also accepts `veya-mcp-server --backend onnx`; its default remains
Torch. ONNX supports the tools below except encoder embedding shortlisting;
`veya_tournament` provides narrowing through model decisions instead.

The server speaks MCP over **stdio**, not HTTP. Configure the client with the console script:

```json
{
  "mcpServers": {
    "veya": {
      "command": "veya-mcp-server",
      "env": {
        "VEYA_DEVICE": "cpu"
      }
    }
  }
}
```

If the client configuration supports a Python executable and arguments, use
`python -m veya.mcp.server` as the equivalent launch form. The client owns the server process;
Veya does not open a network port.

### Available tools

| Tool | What it does | Main inputs |
|---|---|---|
| `veya_status` | Reports the configured or actual device, CUDA availability, loaded checkpoints, preload state, readiness, and package versions. | none |
| `veya_route` | Selects a checkpoint and returns its model, repository, and reason without running a forward pass. | `state`, `questions`, optional `model`, `task`, `lang`, `lang_guess`, `explain` |
| `veya_predict` | Runs typed questions and returns answers, routing metadata, latency, and the answering device when readable. | `state`, `questions`, optional `model` (`auto`, `english`, `multilingual`, or `typed-decisions`), `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `veya_shortlist` | Shortlists a many-option choice question, then answers it and returns the shortlist metadata. | `state`, `questions`, optional `model`, `k` (default `20`), `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `veya_tournament` | Narrows large choices through balanced model elimination, then answers the final question set. | `state`, `questions`, optional `model`, `group_size` (default `16`), `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `veya_predict_long` | Scans overlapping windows and returns each answer with its deciding window's token offsets. | `state`, `questions`, optional `model`, `window`, `stride`, `task`, `lang` |
| `veya_preset` | Runs a built-in workflow using its built-in question set. | `preset`, `state`, optional `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `veya_predict_batch` | Answers many requests in one call. Requests are routed first and grouped by checkpoint, so matching question schemas share forward passes; answers come back in input order. | `requests`, each `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}`, optional `batch_size`, `hooks_timeout`, `min_confidence`, `sort_by_length` |
| `veya_route_batch` | Decides which checkpoint would answer each request, with no forward pass and no checkpoint load. | `requests`, each `{state, questions, model?, task?, lang?, lang_guess?}`, optional `hooks_timeout`, `explain` |
| `veya_decide` | Answers a JSON-schema-shaped decision and returns decided values with per-field confidence. Schema properties may be enum choices, booleans, or integers with a minimum and maximum. Free strings, arrays, and nested objects are rejected by path. | `state`, `schema`, optional `model`, `min_confidence` |

The three batch and schema tools exist because the same operations are available on the SDK and
`veya-serve`: handling many requests, or serving a caller that already knows the answer shape,
does not require dropping to Python. For the schema-driven form in more depth, see
[Schema-driven decisions](structured.md).

The shared guardrail says not to send choice questions with more than 20 options without
narrowing. `veya_shortlist` keeps the `k` most likely labels before the forward pass; its
default is `k=20`. It uses mean-pooled embeddings from the answering checkpoint's own encoder,
so it does not download a second model, and returns the kept labels, cosine scores, `k`,
option count, and whether the question passed through unshortlisted for each shortlisted
question. A question with `k` at or past its option count is passed through unchanged, so
its cosine scores come back `null` rather than empty.

When shortlisting narrows a choice carrying `option_order`, it drops that full-set order and
reports `option_order_dropped` with the original `n` and the number `kept` for each affected
question. Invalid permutations are still refused before prediction. Choices passed through
unchanged keep their order.

`veya_tournament` narrows choices without embeddings: each round evaluates balanced
groups through the decision model, and their winners advance until the finalists fit
`group_size`. It routes the original schema once and pins that checkpoint and language
for every round. The `tournament` metadata records each choice's final `labels`, original
`n`, and elimination `rounds`; small choices report zero rounds. Probabilities and confidence
describe the finalists only, so full-set confidence thresholds need separate validation.
Narrowing reports discarded full-set permutations through `option_order_dropped`.

`veya_predict_long` scans the whole state in overlapping windows. `window` and `stride`
are positive integer token counts; omitted values use the checkpoint budget and 50%
overlap. Noul selects the strongest window, while choice and score select the most
confident window. Each answer includes `window` attribution and the result reports
`windows`, the number scored. This tool accepts `task` and `lang`; token-budget and
confidence-gate controls belong to single-window prediction.

`veya_route` and `veya_route_batch` accept `explain: true` to add `diagnostic` to each
final routing decision. It contains the analysis `stage`, `reason`, and optional `source`
(leaf index, path and 1-based line), without copying the input text. Explicit routing
overrides and hook decisions without detection report `diagnostic: null`. Omit `explain`
to keep the original three-field routing shape. In remote mode these diagnostics describe
the MCP process's local routing preview.

`state` must be a non-empty JSON object. `questions` must be a non-empty object whose values use
Veya's typed question schema. `veya_preset` accepts the same five presets the CLI does: `email`,
`guard`, `moderation`, `triage`, and the router workflow, whose canonical name on this surface is
`model_router`. `router` is accepted as an alias and names the same preset, so the CLI spelling
works here too; the canonical key is the one that comes back in the result. Given a state of
exactly one string, `veya_preset` places it under the field that preset's questions name, the
same placement the CLI does, so a caller does not have to guess the key. Anything richer than one
string is the caller's own shape and is passed through untouched.

Single-window prediction, shortlisting and tournament tools share the batch routing controls. Alongside
`model`, a request may set `task` (name a checkpoint by the work), `lang` (force a language code),
and `lang_guess` (a soft language hint that sits below `lang` and above the built-in detector, so a
probable-but-uncertain code can nudge which checkpoint is chosen without forcing it the way `lang`
does). `lang_guess` only participates in routing, so like `task` it is refused on a call that pins
`model` -- a pinned checkpoint has nothing left to route. `veya_predict` and `veya_shortlist` also
take `max_len`/`head_max_len` for the answering token budget and `min_confidence` for the abstention
gate.

Omitted, `null` and `"auto"` model values all allow automatic routing. For batch tools, omitted
controls inherit the Router configuration; numeric `0` unsets `batch_size` and `hooks_timeout`.
Boolean, negative and non-finite controls are refused, rather than silently discarded.

A prediction call has the same shape as the SDK's typed call:

```json
{
  "state": {
    "body": "I was billed twice for the same plan. Please reverse the duplicate charge."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "payments, invoices, refunds, duplicate charges",
        "technical": "bugs, outages, integration problems"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the user need immediate help?"
    }
  }
}
```

The tool response is JSON containing the typed `answers`, the `routing` decision, and timing
information. Do not treat a high-confidence answer as permission to perform an external action;
the application or agent remains responsible for policy, review, and side effects.

### Startup and environment

The MCP server keeps a resident Router and serializes first-time construction. By default it
preloads `english` and `multilingual`; `typed-decisions` stays lazy. A preload failure is
reported at startup and retried on the next tool call, so inspect `veya_status` before assuming
the server is ready.

| Variable | Default | Meaning |
|---|---|---|
| `VEYA_DEVICE` | automatic | Device preference passed to PyTorch, such as `cpu` or `cuda`. Case is normalized; values PyTorch refuses or rewrites to another device are rejected. |
| `VEYA_BACKEND` | `torch` in full package; `onnx` in lightweight | Local inference backend; `--backend` takes precedence. Lightweight installations support ONNX only. Ignored when `VEYA_BASE_URL` delegates inference. |
| `VEYA_ONNX_PATHS` | unset | JSON map of model names to local exported graphs, paired with local `VEYA_EXTRA_MODELS` checkpoint directories. ONNX preloads configured graph names when `VEYA_MODELS` is omitted. |
| `VEYA_EXTRA_MODELS` | unset | JSON map from checkpoint names to local directories or registered model sources. Send the custom name as `model` to select it explicitly. |
| `VEYA_SHA256_DIGESTS` | unset | Artifact digest map or nested map keyed by checkpoint name. Use per-checkpoint pins when loading more than one checkpoint. |
| `VEYA_PERFORMANCE_MODE` | unset | Opt-in resource profile `1`–`5`, or `default` for existing defaults. Sets thread, batch, resident-model and preload budgets for local inference. Explicit existing settings override profile values. |
| `VEYA_MAX_LOADED` | Router default | Maximum resident checkpoints. Overrides the profile's model cap. |
| `VEYA_PRELOAD` | `1` | Build the configured checkpoints at startup. Set to `0` for lazy loading. |
| `VEYA_MODELS` | `english,multilingual` | Comma-separated checkpoints to preload. An empty value keeps the MCP default rather than preloading every checkpoint. |
| `VEYA_THREADS` | PyTorch default | Caps Torch intra-op threads for CPU inference; keep it at or below the physical core count. |
| `VEYA_AUTO_TASK` | `0` | Set to `1` to let a request auto-route to the `typed-decisions` checkpoint. Same meaning as in `veya.serve`; it does not preload that checkpoint, so `VEYA_MODELS` still decides what is built at startup. |
| `VEYA_DEFAULT_MODEL` | `english` | The checkpoint a state with no language evidence falls back to, same meaning as in `veya.serve`. Unlike `veya.serve`, an unresolvable name does not stop the server: it comes back as a `router construction failed` tool error on the next call, because a stdio server has no startup to refuse. |
| `VEYA_CACHE_PATH` | unset | Local SQLite decision file; a nonempty path enables persistent replay. |
| `VEYA_CACHE_TTL` | `86400` | Retention in seconds, or `none` for no expiry. |
| `VEYA_CACHE_MAXSIZE` | `10000` | Decision count target, or `none` for no capacity eviction. |
| `VEYA_CACHE_FINGERPRINT` | unset | Explicit policy identity across model upgrades; empty uses automatic model/configuration identity. |
| `VEYA_CACHE_REQUIRED` | `0` | Require successful replay or commit; storage failures become tool errors. |
| `VEYA_CACHE_REQUIRE_EXISTING` | `0` | Refuse to create a missing SQLite decision database. |
| `VEYA_CACHE_EXPECTED_STORE_ID` | unset | Require an existing decision store with the approved UUID. |
| `VEYA_REPLAY_MANIFEST` | unset | Verified local metadata for replay without loading model weights. |
| `VEYA_DECISION_POLICY` | checkpoint default | Select `canonical` or [`context-balanced-v1`](decision-policy.md) on the local backend. |
| `VEYA_MCP_MODE` | `classic` | Select the existing tool catalog or the opt-in local JavaScript `code` interface; `--mode` takes precedence. |
| `VEYA_BASE_URL` | unset | Send predictions to a `veya-serve` on your own hardware instead of loading checkpoints in each MCP process. A bare `host:port` is read as HTTP. |
| `VEYA_REMOTE_TIMEOUT` | `300` | HTTP timeout in seconds when `VEYA_BASE_URL` is set, including the server's cold load. Invalid or non-positive values use the default. |
| `VEYA_USER_AGENT` | `veya/<version>` | HTTP client identity in remote mode. Empty values use the default; invalid header characters are refused. Custom `RemoteRouter(user_agent=...)` overrides the environment. |

`veya_status` includes `performance_mode` and `performance_config` (both null without a profile).
`checkpoint_policies` reports the effective policy of each resident local checkpoint.
`model_free_replay.enabled` reports whether replay metadata is configured. Cold Torch
status reports the configured device preference without probing Torch when replay is configured.
The configuration reports effective budgets and overrides. Torch thread changes affect the
whole process; choose the mode at startup. The [mode guide](performance-modes.md) lists all five
budgets, Python examples and the limits of the measurements.

The local MCP builder uses the same [decision retention settings](production.md#cache-configuration)
as HTTP. For durable replay, combine a persistent `VEYA_CACHE_PATH`, `VEYA_CACHE_TTL=none`,
`VEYA_CACHE_MAXSIZE=none` and an explicit policy identity such as `policy-v1`.
When `VEYA_BASE_URL` is set, configure these values in the HTTP server's environment.
Add `VEYA_CACHE_REQUIRED=1` to refuse successful predictions when storage fails;
local tools return `decision_cache_unavailable`. Optional mode still permits recomputation.
Neither mode can recover decisions from a lost database.

Use [verified replay](replay.md) to approve the store identity and policy metadata before
serving retained hits without loading model weights. The optional
[MCP Code Mode](mcp-code-mode.md) composes validated decisions in bounded local JavaScript;
install `veya[codemode]` and select `veya-mcp-server --mode code` to use it.

### Share one model server across MCP sessions

Run one local HTTP server and point each MCP client's environment at it:

```bash
VEYA_HOST=127.0.0.1 VEYA_PRELOAD=0 VEYA_IDLE_UNLOAD_SECONDS=300 veya-serve
```

```json
{
  "mcpServers": {
    "veya": {
      "command": "veya-mcp-server",
      "env": {"VEYA_BASE_URL": "http://127.0.0.1:8000"}
    }
  }
}
```

Install `veya[serve]` where the HTTP server runs. MCP still uses stdio with the editor; its
prediction tools use HTTP to reach your server. `veya_predict`, `veya_predict_batch`, `veya_decide`
and `veya_preset` use the server's original state, instructions and option descriptions.
Heterogeneous batches send one `/v1/systemone` request per item, preserving input order;
`batch_size` and `sort_by_length` do not change the server's execution. `veya_status` reports the
server's `/health`; `veya_route` and `veya_route_batch` stay local and need no model or HTTP request.
The MCP process imports no torch and loads no checkpoint, including when `VEYA_THREADS` or
`VEYA_PRELOAD` is set.

In remote mode, status reports the HTTP server's `performance_mode` and `performance_config`.
Set `VEYA_PERFORMANCE_MODE` on that server; a profile in the MCP process does not change
remote resources.

Set the same `VEYA_API_KEY` in both processes when the server requires a bearer token. Keep
`VEYA_DEFAULT_MODEL` and `VEYA_AUTO_TASK` aligned so local routing previews match the server's
actual routing. Device and preload settings belong to the HTTP server. The first call after an
idle unload waits for a cold load; raise `VEYA_REMOTE_TIMEOUT` if that takes longer than 300 seconds.
`veya_shortlist`, `veya_tournament`, `veya_predict_long` and prediction hook overrides return
`unsupported_remote` before making HTTP requests, since their code needs the model process.
For server-side tournament decisions use `veya_predict` with `VEYA_TOURNAMENT_GROUP` enabled
on that self-hosted server. HTTP errors retain the server's detail text as MCP tool errors. With
`VEYA_BASE_URL` unset, the MCP server keeps loading checkpoints in its own process.

The stock `veya-mcp-server` launcher creates its Router without installing hooks. Install prediction
hooks in the process that runs inference: the MCP process in local mode, or the HTTP server in
shared-server mode. A custom launcher can use `veya.hooks.set_default_hooks` before building its Router. The environment variables
above configure model lifecycle, not hook registration. The client still decides when to call a
tool and what to do with the returned decision.

## 3. Shared boundaries and related guides

The CLI and MCP server are interfaces to the same typed decision engine:

- Use `choice` for a finite label set, `score` for an ordered rubric, and `noul` for the
  probability of true.
- Validate thresholds and presets on representative data; there is no universal adoption
  threshold.
- Keep irreversible or high-cost actions behind the application's review and fallback policy.
- The MCP server calls `Router.predict`, so hooks fire when a custom launcher installs them. See
  [Prediction hooks](hooks/index.md), [hook lifecycle](hooks/lifecycle.md), and
  [Tracing](hooks/tracing.md) for observability and `run_id` correlation.

This guide covers the local CLI and the built-in MCP stdio server. It does not document the
HTTP API, community wrappers, or an MCP protocol redesign.
