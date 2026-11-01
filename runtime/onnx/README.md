# Laya-Pro ONNX runtime

Local decision inference using ONNX Runtime, NumPy and Hugging Face Tokenizers.
The package has no PyTorch, Transformers or ONNX exporter dependency.

Build or install from the Laya-Pro checkout:

```bash
pip install ./runtime/onnx
```

```python
from laya_onnx import ONNXAgent

agent = ONNXAgent("/path/to/checkpoint", onnx_path="/path/to/model.onnx", performance_mode=2)
result = agent.predict(
    {"message": "Please refund the duplicate charge"},
    {"team": {"type": "choice", "instructions": "Which team should respond?",
              "criteria": {"billing": "Payments and invoices", "support": "Technical help"}}},
)
```

The checkpoint must contain `rl_agent_config.json` and a saved fast tokenizer under
`tokenizer/tokenizer.json`, with its special-token metadata. The graph must be exported
from that checkpoint. Export is a separate step using the full project's
`scripts/export_onnx.py`; model weights and graphs are not included in this package.

Single, batch and long-window predictions, ordered typed options, schema decisions,
calibration loading, hooks, decision-cache retention and performance modes use the same
inference implementation as Laya-Pro. Native tokenization requires saved tokenizer JSON;
unsupported or missing tokenizer metadata is an error rather than a fallback to another
tokenizer. The full package retains its existing Transformers tokenizer by default.

`Agent` is an alias for `ONNXAgent`; `load(...)` constructs an ONNXAgent. `DecisionCache`,
hook types, `decide` and `decide_batch` are available from `laya_onnx`. JSON schemas need
no extra dependency; install `.[structured]` here to use Pydantic model schemas.

HTTP and local MCP services are optional:

```bash
python -m pip install './runtime/onnx[serve]'  # FastAPI and uvicorn, from the repository root
python -m pip install './runtime/onnx[mcp]'    # MCP stdio protocol, from the repository root
```

Configure local checkpoint directories and their exported graphs, then start either service:

```bash
export LAYA_EXTRA_MODELS='{"english":"/path/to/checkpoint"}'
export LAYA_ONNX_PATHS='{"english":"/path/to/model.onnx"}'
export LAYA_MODELS=english
laya-onnx-serve --performance-mode 2
# Equivalent: python -m laya_onnx.serve
laya-onnx-mcp --performance-mode 2
# Equivalent: python -m laya_onnx.mcp.server
```

These commands default to ONNX. `LAYA_BACKEND` and `--backend` select the runtime,
with the command-line flag taking precedence; this distribution supports `onnx`
only. Services use the CPU provider; set `LAYA_DEVICE=cpu` or leave it unset.
Other service device types are rejected.
The canonical HTTP authentication, validation, routing, batch inference, cache
retention and idle unloading apply here too. MCP uses the same local router and
tools; encoder embedding shortlisting is unavailable for exported decision graphs.
The base install adds neither HTTP nor MCP dependencies.

The `laya_onnx` namespace is separate from `laya`, so both distributions can be installed
together. Their runtime code is copied from the canonical repository modules at build
time. Source distributions carry those inputs and can be rebuilt outside the checkout.
This change does not publish the distribution to PyPI.

See the [runtime guide](https://tgundhus.github.io/laya-pro/onnx-runtime/) for compatibility,
benchmarks and the distinction between inference and export dependencies.
