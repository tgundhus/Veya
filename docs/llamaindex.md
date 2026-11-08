# LlamaIndex Integration

Veya provides local decision components for **LlamaIndex** RAG pipelines,
`RouterQueryEngine` and tool selection. Latency depends on hardware, checkpoint and questions;
see the [current review](reports/production-review.md) and [archived benchmarks](benchmarks.md).

* **`VeyaSingleSelector`**: Single-choice selector replacing `LLMSingleSelector` for `RouterQueryEngine`.
* **`VeyaMultiSelector`**: Multi-choice selector replacing `LLMMultiSelector` for composite queries spanning multiple data sources.
* **`VeyaQueryRouter`**: Standalone query dispatcher routing incoming requests directly to target query engines or callables.

Supports both **local in-process inference** (`Agent` or `Router`) and **remote HTTP inference** against your own `veya-serve` instance without requiring PyTorch on edge clients.

---

## Installation

```bash
pip install "veya[llamaindex] @ git+https://github.com/tgundhus/Veya.git"
```

---

## 1. Single-Choice Routing with `RouterQueryEngine`

In LlamaIndex, `RouterQueryEngine` uses a selector to choose the query engine or tool for a
question. `VeyaSingleSelector` evaluates those candidates without generating text:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from veya.integrations.llamaindex import VeyaSingleSelector

# Define query engine tools
docs_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_documentation",
        description="Semantic search over technical user documentation and API guides.",
    ),
)
sql_tool = QueryEngineTool(
    query_engine=sql_index.as_query_engine(),
    metadata=ToolMetadata(
        name="sql_database",
        description="Structured SQL database containing customer accounts, billing, and orders.",
    ),
)

# Initialize the selector with confidence fallback
selector = VeyaSingleSelector(
    confidence_threshold=0.80,   # If confidence < 0.80, fall back to index 0
    fallback_index=0,
)

router_engine = RouterQueryEngine(
    selector=selector,
    query_engine_tools=[docs_tool, sql_tool],
)

response = router_engine.query("What is the shipping address for order #4912?")
print(response)
```

---

## 2. Multi-Choice Selection for Composite Queries

For queries that require synthesis across multiple indexes (e.g. comparing documentation specs with transactional database records), `VeyaMultiSelector` evaluates candidate relevance and returns multiple selected tools:

```python
from veya.integrations.llamaindex import VeyaMultiSelector

multi_selector = VeyaMultiSelector(
    probability_threshold=0.25,  # Select all tools with probability >= 0.25
    max_outputs=2,
)

tools = [docs_tool.metadata, sql_tool.metadata, summary_tool.metadata]
result = multi_selector.select(
    tools,
    "How does the database security policy compare with our published compliance guide?"
)

for sel in result.selections:
    print(f"Tool: {tools[sel.index].name} | {sel.reason}")
```

---

## 3. Direct Query Dispatch with `VeyaQueryRouter`

For direct routing without the overhead of `RouterQueryEngine`, `VeyaQueryRouter` routes queries directly to dictionary-registered engines:

```python
from veya.integrations.llamaindex import VeyaQueryRouter

router = VeyaQueryRouter(
    query_engines={
        "vector": vector_query_engine,
        "sql": sql_query_engine,
        "summary": summary_query_engine,
    },
    descriptions={
        "vector": "Semantic search over product documentation and guides",
        "sql": "Structured SQL queries for user accounts and transactions",
        "summary": "Quarterly reports and high-level business summaries",
    },
    confidence_threshold=0.75,
    fallback_key="vector",
)

# Route and execute in one call:
response = router.query("How many active subscriptions were renewed in Q3?")
print(response)
```

Both synchronous `query()` and asynchronous `aquery()` are supported.

---

## 4. Confidence Threshold Gating

Like Veya's LangChain integration, `VeyaSingleSelector` and `VeyaQueryRouter` read calibrated `answer_confidence` (`max(p)`):

- **Automatic Fallback:** Specify `fallback_index` (or `fallback_key`) to seamlessly divert uncertain queries to a safe default engine.
- **Strict Guarding:** Set `raise_on_low_confidence=True` on `VeyaSingleSelector` to raise `VeyaLowConfidenceError` when input is ambiguous, allowing caller escalation.

---

## 5. Remote HTTP Deployments

For serverless RAG, edge environments, or environments without local GPUs:

```python
from veya.integrations.llamaindex import VeyaSingleSelector

selector = VeyaSingleSelector(
    base_url="http://veya-serve.internal:8080",
    confidence_threshold=0.85,
    fallback_index=0,
)
```

The remote client uses Python's standard library `urllib` with zero heavy dependencies, preventing cross-origin credential forwarding and matching the `/v1/systemone` specification.

---

## 6. Per-call decision controls

`VeyaSingleSelector`, `VeyaMultiSelector` and `VeyaQueryRouter` take the same per-call arguments the
core API does: the two token budgets (`max_len`, `head_max_len`), the language and abstention
controls (`lang`, `min_confidence`), and the five prediction-hook
arguments (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`). They are
per selector, so a wide routing step can be given room while the rest of the pipeline keeps the
checkpoint's defaults.

A choice question's options share the checkpoint's *option* budget -- `head_max_len`, 192 tokens on
`veya` -- and every candidate contributes its name and description, so past roughly 20 tools the
descriptions start reaching the model as the same text.

```python
selector = VeyaSingleSelector(
    instructions="Which tool or query engine is best suited to answer this query?",
    max_len=1024,          # total window
    head_max_len=512,      # tokens shared by the option prompt
)
result = selector.select(tools, query)     # tools: 59 descriptions
```

Measured on `veya` (Apple silicon, one forward pass per query, scored on the chosen tool) with a
59-tool roster built from the MASSIVE en intent labels and one utterance per label, so ground truth
is exact. Each cell is how many of the 59 queries reached their own tool; both repeats gave the
same count.

| 59-tool roster | Default budget | `max_len=1024, head_max_len=384` | `…, head_max_len=512` |
|---|---|---|---|
| Queries on their own tool | 2/59 | 8/59 | 15/59 |
| Median ms per query | 160 | 172 | 184 |

Absolute accuracy is not the claim here: the checkpoint is not a MASSIVE classifier, and 59 similar
labels are a stress shape. The claim is the direction and the price -- a roster the default budget
collapses to near-nothing is readable, and at this size the window costs little time. With fewer
than about 20 options the labels already fit and widening can move answers the wrong way, which is
why both arguments are opt-in per selector. See the [LangChain
integration](langchain.md#7-widening-the-token-budget-for-many-options) for that measured cliff.

**Hooks run on the local path only.** A selector with a `base_url` and `hooks=[...]` raises
`ValueError` rather than reporting a success whose cache never ran -- a hook is a Python callable
that runs inside `predict`, and no wire format carries it. Install hooks in the process that runs
inference. The two budgets do travel to a remote node, in the request body, up to its
`VEYA_MAX_TOKEN_BUDGET` ceiling; a larger value comes back as a 422.

### Language and abstention

`lang` pins the language the query is routed and answered in -- selecting the answering
checkpoint's per-language calibration instead of relying on built-in detection -- and
`min_confidence` is core's abstention gate: a decision under it comes back as an abstention rather
than a forced selection. Both are read by `Agent.predict` and `Router.predict` alike and accepted by
`veya-serve` in the request body, so a selector forwards them on the local and the remote path. An
unset one is omitted, not sent as `None`, so it cannot shadow the deployment's own default;
`min_confidence=0.0` and `lang=""` are real values and are forwarded as given.

```python
selector = VeyaSingleSelector(
    instructions="Which tool or query engine is best suited to answer this query?",
    lang="de",             # answer German queries in German
    min_confidence=0.3,    # abstain when no tool clears a 0.3 confidence
)
```
