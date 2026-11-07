---
title: Python and TypeScript routing contract
description: Versioned Unicode, JSON traversal, language evidence and checkpoint routing shared by the local Python and native TypeScript SDKs.
---

# Shared routing contract

Python and the native TypeScript SDK share the versioned
`veya-routing-v1-unicode15` routing contract. For the same supported state, ordered
questions, built-in model registry and routing controls, they return the same
checkpoint selection, language evidence, diagnostic source and human-readable
route reason. Python result keys use snake_case; TypeScript uses camelCase.

```python
from veya import Router
from veya.lang import ROUTING_CONTRACT_VERSION

decision = Router().route(state, questions, explain=True)
print(ROUTING_CONTRACT_VERSION, decision["model"], decision["detection"])
```

```typescript
import { Router, ROUTING_CONTRACT_VERSION } from "veya-ts";

const decision = new Router().route(state, questions, { explain: true });
console.log(ROUTING_CONTRACT_VERSION, decision.model, decision.detection);
```

`analyse(state, { explain: true })` and `Router.routeBatch(..., { explain: true })`
also expose TypeScript diagnostics. Explanation is opt-in and does not change the
selected checkpoint. Explicit model/task/language controls skip language detection,
so those route results keep `detection: null`.

## Supported inputs and ordering

The cross-SDK contract covers parsed JSON values: strings, arrays, objects with
string keys, finite numbers, booleans and null. Only string leaves contribute
language evidence. Object keys are ignored as text. Arrays retain their order.
The same supported string model/task/language controls and Router defaults must
be supplied in both SDKs. Custom routing functions, hooks, registries and native
object classes remain application-defined behavior.

State objects follow ECMAScript property enumeration: canonical decimal index keys
from `"0"` through `"4294967294"` come first in numeric order, then other keys in
insertion order. `"01"`, `"-1"` and `"4294967295"` are ordinary keys. Python applies
that rule to state traversal so a JSON object has the same language evidence in
both runtimes. It does not reorder the caller's object or any question/criterion
definition. Use arrays when state field order is itself meaningful.

State-text budgets in this contract are integers; nonpositive budgets return empty
text. The default joined state-text budget is 4,000 Unicode code points, including joining
spaces. Combining marks count separately; surrogate pairs for an astral character
count once. Nested traversal stops beyond depth six. The segment pass shares a
4,000-code-point budget, while the structured-value pass independently examines
up to 4,000 code points of each line. A long earlier field or line therefore cannot
hide a later foreign-language value. Values beyond an exhausted state-text budget
are not read; canonical ordering still requires enumerating mapping keys.

The contract preserves text rather than applying Unicode normalization or treating
similar inputs as the same decision. JSON parsers combine paired UTF-16 escapes
into their Unicode character. Native Python strings containing literal paired
surrogate code points should be JSON-normalized before cross-SDK comparison.
Isolated surrogate code points are covered by the local routing fixtures; the
HTTP interface retains its stricter input validation.

Python bytes/bytearray and TypeScript Uint8Array additionally support strict UTF-8
decoding. A UTF-8 BOM is preserved as a character; invalid UTF-8 contributes no
text. These binary extensions are tested separately from JSON interoperability.

## Frozen Unicode semantics

Classification, token boundaries, casing, combining classes, whitespace and
printable route evidence use Unicode 15.0.0 semantics. Checked-in generated
tables make routing independent of the host Python or JavaScript Unicode release.
Contextual Greek final sigma and the Turkish dotted capital I follow the Python
reference rules. Python uses native character operations when its Unicode database
is already 15.0.0 and the equivalent frozen implementation otherwise.

The Unicode table SHA-256 is
`1acf1e0ec0edd3e7a2fe90e2f2a27d2f39de0ab52528ccb2143ee485274d3bba`.
Regenerate or validate the tables with Python 3.12 / Unicode 15.0.0:

```bash
python research/scripts/generate_routing_unicode.py --check --check-node
python research/scripts/generate_routing_corpus.py --check
python tests/test_routing_parity.py
cd veya-ts && npm test -- --run tests/routing-parity.test.ts
```

The Unicode validator compares all 1,114,112 code points and 10,000 contextual
casing examples. The 101-case shared golden corpus checks flattening budgets, complete
analysis, explanatory evidence, profiles and full route results. A fixed-seed
test additionally compares 3,000 live generated cases and the canonical stopword
vocabulary across Python and compiled TypeScript. A separate vocabulary guard
prevents the ports' stopword/diacritic definitions from drifting. The optional live
check needs Node and the local TypeScript development
toolchain; both SDKs independently test the recorded golden expectations.

## Migration and limits

This contract intentionally corrects Python's former insertion order for numeric
state keys and freezes classification across Python Unicode versions. It also
removes TypeScript-only English stopwords and Vietnamese diacritic additions that
were absent from the canonical Python implementation. Some previous TypeScript
language guesses, configured-default routes and Vietnamese automatic routes can
therefore change. For known Vietnamese input, supply `lang="vi"` / `{ lang: "vi" }`;
the shared heuristic does not contain a Vietnamese stopword model and cannot
reliably infer every Vietnamese sentence.

Use the new routing contract in persisted policy identities and regenerate replay
manifests when adopting the change. An explicit cache fingerprint can deliberately
retain an old decision; checkpoint routing remains part of the key, so migration
must account for any changed selected route. [Decision retention](consistency.md)
describes those guarantees. Matching routing does not establish identical numerical
inference across native SDKs, precisions or backends, or the correctness of an answer.

## Measurements

The [routing-contract benchmark](https://github.com/tgundhus/Veya/blob/main/research/scripts/bench_routing_contract.py)
compares the previous reviewed source (`6496fc7a`) with the new Python and TypeScript
preprocessing. It records alternating raw timings, source identities and output
changes, without loading weights or measuring model inference. Canonical numeric-key
ordering adds a necessary key-enumeration cost to unusually wide Python mappings;
ordinary strings, bytes and nested lists retain bounded value traversal.

The [recorded Windows run](https://github.com/tgundhus/Veya/blob/main/research/results/routing_contract_windows_20261105.json)
used nine alternating samples. Values below are median per-call preprocessing times;
the workstation load was uncontrolled.

| Workload | Python analysis before → after | TypeScript analysis before → after |
|---|---:|---:|
| Short English string | 10.487 → 10.421 microseconds | 6.343 → 6.465 microseconds |
| Support ticket | 274.021 → 281.086 microseconds | 92.986 → 104.612 microseconds |
| Long first field, 10,000 tail keys | 3.069 → 5.192 milliseconds | 12.010 → 4.208 milliseconds |
| 10,000 numeric state keys | 4.308 → 19.635 milliseconds | 10.533 → 3.096 milliseconds |

For the wide first-field case, Python state-text flattening alone increased from
1.150 to 493.125 microseconds because it must enumerate keys before choosing the
first field. For 10,000 numeric keys, flattening increased from 1.762 to 2,814.188
microseconds and its output intentionally changed to canonical index order. Those
wide-map costs are a correctness tradeoff, not a speed improvement. TypeScript's
per-line short-value guard reduces work on wide maps; its ordinary support-ticket
analysis became slightly slower. The after-emoji-line fixture also intentionally
changes TypeScript's verdict to match Python's independent line budget.

```bash
python research/scripts/bench_routing_contract.py --out research/results/routing_contract.json
```

The optional TypeScript benchmark requires Node 22.18+ and the installed local SDK
development toolchain. Use `--skip-node` for a Python-only comparison. No runtime
feature downloads Unicode data or invokes a hosted service.
