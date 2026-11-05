---
title: Local MCP Code Mode
description: Compose validated Veya operations in bounded local JavaScript, with persistent models, authoritative replay and no hosted execution dependency.
---

# Local MCP Code Mode

Code Mode adds programmable composition to the existing MCP stdio server. The calling
agent writes a JavaScript function; Veya continues to answer typed decision questions.
One outer tool call can batch inputs, inspect answers, make dependent decisions and
return a compact JSON result. Every inference call uses the existing validated MCP
functions and the server's resident Router and configured decision cache.

The default remains the classic ten-tool catalog. Code Mode is optional and fully
local, including its QuickJS-NG runtime and process CPU monitor. No Cloudflare account,
hosted sandbox, Node.js installation or external execution API is required.

## Install and start

```bash
python -m pip install "veya[codemode] @ git+https://github.com/tgundhus/Veya.git"
laya-mcp-server --mode code
```

The lightweight ONNX distribution also supports its `codemode` extra and
`laya-onnx-mcp --mode code`. Core ONNX inference still needs neither Torch nor
Transformers. Pin a reviewed source revision for deployment.

`LAYA_MCP_MODE=code` selects the same mode. Valid values are `classic` and `code`;
`--mode` takes precedence. Invalid values fail startup. Resource flags apply only
to code mode and reject invalid or nonpositive values before model loading:

```bash
laya-mcp-server --mode code --code-wall-seconds 120 --code-cpu-seconds 2 \
  --code-memory-mib 64 --code-max-calls 32
```

An MCP client configuration can select the single-tool interface:

```json
{
  "mcpServers": {
    "veya": {
      "command": "laya-mcp-server",
      "args": ["--mode", "code"],
      "env": {"LAYA_PRELOAD": "0", "LAYA_DEVICE": "cpu"}
    }
  }
}
```

Existing model, backend, cache and verified replay settings keep their meanings.
For persistent authoritative decisions, configure [required storage](consistency.md)
and an explicit policy identity. [Verified replay](replay.md) can answer retained
hits without loading weights; a miss loads and validates the selected checkpoint's
actual work budget before a forward pass.

## Single-tool API

The code catalog advertises only `veya_execute(code, input)`. Its description includes
the question grammar, result envelopes, a composition example and method argument
names/requiredness generated from the existing operation signatures.

`code` is a JavaScript function expression, usually `async (input, veya) => {...}`.
The separate `input` argument contains JSON data. The same frozen `veya` API is also
available as a global. Calls take one named-argument object:

```javascript
async (input, veya) => {
  const questions = {
    urgent: {type: "noul", instructions: "Does this need attention today?"}
  };
  const batch = await veya.predict_batch({
    requests: input.map(state => ({state, questions}))
  });
  return batch.requests.map((result, index) => ({
    index,
    needsAttention: result.answers.urgent.noul >= 0.8
  }));
}
```

For that example, `input` can be `[{"message":"The service is unavailable"},
{"message":"Please send last month's invoice"}]`.

Available methods are `predict`, `predict_batch`, `route`, `route_batch`, `shortlist`,
`tournament`, `predict_long`, `preset`, `decide` and `status`. Their validators,
decision fields and backend limitations match the existing [MCP tools](cli-mcp.md).
For example, ONNX does not export the embeddings needed by `shortlist`; its normal
unsupported-backend error remains available to the script. Host-only arguments,
including `router`, `agent`, callable embedding functions and hook overrides, are
never accepted from JavaScript. Contradictory pinned-model routing hints are refused
before model loading, including in a code-mode batch.

Await every operation. Fire-and-forget calls and unresolved promises fail the workflow.
`Promise.all` can compose promises, but host operations execute serially to protect
shared inference state; use `predict_batch` for independent inputs. Return a finite
JSON value. Returning `undefined`, functions, cyclic objects or nonfinite numbers fails.

## JSON and decision identity

Data crosses the guest boundary as bounded JSON bytes. No Python object or callable
enters the guest. JavaScript has one numeric type: input integers must be within
the exact safe range, and Python callers should use integers for whole-number inputs
rather than `1.0` or `-0.0`. Strings can preserve formatting-sensitive numeric text.

ECMAScript enumerates array-index object keys first in ascending order. Inputs and
host replies whose mapping order would change are refused, rather than silently
changing question order, positional choice criteria or cache identity. For example,
criteria ordered as `"10"`, then `"2"` need an explicitly compatible label/order
representation before using Code Mode. Ordinary descriptive string labels are unchanged.

Each queued operation snapshots its arguments immediately. Later JavaScript mutations
cannot rewrite an already queued decision. The decision cache remains authoritative
for each prediction. The final script result is a caller-defined transformation;
there is no implicit cache of complete workflow outputs or arbitrary script logic.

## Resource and failure boundaries

Each workflow gets a fresh QuickJS context in a disposable spawned process. The
embedded engine has no operating-system standard library, Python callbacks, module
loader, filesystem, network, environment or shell bindings. Raw microtask and
finalization callback APIs are disabled; Promise-based composition remains available.
The Promise constructor and prototype remain frozen so guest expressions cannot
replace the runtime's internal continuation machinery.
The guest receives only the validated operation capabilities.

| Default limit | Value |
|---|---:|
| JavaScript CPU time, cumulative across jobs and RPC resumes | 2 seconds |
| Workflow wall time after input validation/admission, including startup and host inference | 120 seconds |
| JavaScript heap / stack | 64 MiB / 512 KiB |
| Source size | 64 KiB |
| Input / each operation argument / each result | 1 MiB each |
| Total encoded workflow traffic | 8 MiB |
| Host operations / guest promise jobs | 32 / 10,000 |
| JSON nesting / item count | 32 / 50,000 |
| States / questions / total options per operation | 64 / 64 / 512 |
| Choice options / tournament or shortlist input options / score levels | 100 / 255 / 32 |
| Token width / aggregate forward-token work per operation | 8,192 / 131,072 |
| Serialized state characters / scan windows | 50,000 / 64 |

Host resource checks cover schema-generated questions, preset questions, shortlist
embeddings, tournament rounds, actual resident checkpoint widths and exact tokenizer
scan windows. They run before inference, with raw bounds enforced before cold model
loading. Code Mode accepts the built-in Router and checkpoint implementations with
cache-only hooks and declarative routing. Arbitrary custom hooks or Agent classes
can rewrite validated work and therefore require the classic trusted-host interface.
Remote mode retains the remote server's own inference limits and uses conservative
work estimates where resident metadata is unavailable.

QuickJS interrupts CPU work, and an independent parent process monitor also checks
worker CPU roughly every 10 milliseconds. This covers native exception-conversion
paths; operating-system clock and scheduling granularity allow small overshoot.
The wall deadline covers guest work even when it does not consume CPU.

Only one workflow is admitted per runner. A wall timeout or interruption terminates
the guest and prevents more operations. Already-running model inference cannot be
safely interrupted; it may finish and commit its decision afterward. Its admission
permit stays occupied until actual completion, so retries cannot accumulate abandoned
host jobs. Graceful shutdown drains that job before closing its Router/cache.
Concurrent `close()` stops and waits for guest cleanup. Call it from outside the
runner's active caller or host-operation callback; reentrant close raises `RuntimeError`
instead of waiting on itself.

Required-cache errors abort execution outside JavaScript's catchable error flow.
A `try/catch` cannot turn unavailable authoritative storage into a replacement answer.
Remote transport failures are also terminal because a timed-out remote operation may
already have committed. Ordinary validated request errors can be caught and inspected
through their `code` property. Unexpected host exceptions return fixed safe errors.

This is language capability isolation plus process crash/deadline isolation. It does
not claim an OS security boundary against exploitation of a native engine defect.
Keep the optional runtime patched and run the service with appropriate OS permissions.

## Python embedding and verification

The lightweight distribution exposes the same classes through
`laya_onnx.mcp.code_mode`; install `veya-onnx[codemode]` for that interface.

```python
from laya.mcp.code_mode import CodeModeLimits, CodeModeRunner

runner = CodeModeRunner(CodeModeLimits(wall_seconds=60, max_calls=16))
try:
    result = runner.execute("async input => (await veya.predict(input)).answers",
                            {"state": {"message": "Service unavailable"},
                             "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}},
                            router=router)
finally:
    runner.close()
```

`create_code_server()` owns and closes its runner through the MCP lifespan. When
passed an explicit runner, its caller owns that runner's lifetime.

Run `python -m pytest tests/test_mcp_code_mode.py tests/test_mcp_code_budget.py -q`
for resource, validation, required-cache, cancellation, capability and lifecycle
regressions. Installed-wheel tests also exercise the real stdio MCP transport.
The benchmark harness compares equivalent classic and code workflows against an
existing local checkpoint; it reports sandbox overhead, outer-call counts, actual
forward calls and compact response bytes. Reducing agent round trips does not imply
faster encoder inference, and no hosted-LLM latency or token saving is assumed.
