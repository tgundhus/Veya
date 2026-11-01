# Local ONNX service measurements

Real localhost HTTP and MCP stdio; one pinned English checkpoint, fp32 CPU, 4 threads. Cache disabled for first/warm single/batch timings. Filesystem caches were left intact. All variants use the same interpreter and dependency versions. Lightweight distribution installation is verified separately; this is a controlled runtime comparison.

| transport | service | start to first answer s | first request ms | warm single ms | warm batch (4 states) ms | cache hit ms | peak WS MiB | peak private commit MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| http | baseline-torch | 5.671 | 338.75 | 279.28 | 1739.57 | 0.63 | 2060.8 | 4233.1 |
| http | full-onnx | 2.532 | 277.05 | 263.97 | 1672.27 | 0.69 | 1863.3 | 2128.0 |
| http | lite-onnx | 2.593 | 280.11 | 266.89 | 1669.84 | 0.71 | 1863.2 | 2127.9 |
| mcp | baseline-torch | 5.452 | 303.02 | 287.59 | 1760.19 | 0.82 | 2094.4 | 4246.7 |
| mcp | full-onnx | 2.434 | 284.87 | 266.35 | 1629.43 | 0.87 | 1880.3 | 2142.2 |
| mcp | lite-onnx | 2.426 | 274.29 | 265.49 | 1690.17 | 0.83 | 1880.5 | 2142.3 |

Medians across 3 fresh processes per row; warm calls pool 5 samples per process. Startup includes Python, imports, preload, protocol readiness/handshake and the first answer. Peak counters cover the entire server lifetime through untimed parity and cache checks. Private commit excludes shared file mappings; working set includes resident mapped pages.

| transport | service | outside Router, warm single ms | outside Router, warm batch ms | outside Router, cache hit ms |
|---|---|---:|---:|---:|
| http | baseline-torch | 0.917 | 0.997 | 0.525 |
| http | full-onnx | 0.895 | 1.048 | 0.589 |
| http | lite-onnx | 0.866 | 1.121 | 0.596 |
| mcp | baseline-torch | 1.149 | 1.297 | 0.689 |
| mcp | full-onnx | 1.140 | 1.295 | 0.741 |
| mcp | lite-onnx | 1.075 | 1.251 | 0.700 |

Outside-Router time is client round-trip minus an instrumented Router call: it includes request serialization, protocol/framework validation and response serialization, plus scheduling. It is not a network-only measurement. Router wrappers add two perf_counter reads and one list append; raw logit capture is enabled only on untimed parity calls.

Shared typed-answer labels and schemas agree across transports/variants; the HTTP-only x_jev_confidence extension is excluded when comparing HTTP with MCP. Maximum absolute differences: public answer numbers 0, raw logits 1.32918358e-05, unrounded probabilities 8.94069672e-07, action probabilities 0. Declared absolute tolerances: public answers 0.0002 (four-decimal wire rounding), raw logits 0.0001, unrounded probabilities/actions 2e-05. Four hand-written states cover choice, score and yes/no; this is numerical parity, not an accuracy evaluation.

The full-package and standalone ONNX services execute the same canonical native-tokenizer Router/HTTP/MCP code and graph. Their comparison controls runtime/backend changes. The frozen main service uses the default Torch backend. Mixed F16/F32 checkpoint weights are cast by the Torch CPU loader to fp32; the exported graph has FLOAT floating initializers and outputs. Loaded Torch dtype/no-AMP and ORT CPU provider/output types/thread count are asserted. Cache hit measurements enable the in-memory DecisionCache after cache-off tests; its counters assert one miss and the requested hits. The checkpoint remains resident during cache timing.

Baseline commit: `e823ea169941193b83b91322ecd01b1d21f32a83`. Raw requests/responses, process order, source/artifact/wheel/harness hashes, Windows stage counters, versions, stderr and the command are in the adjacent JSON. The workstation was otherwise uncontrolled; small differences are timing noise.
