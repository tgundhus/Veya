# veya-client

A dependency-free HTTP client for self-hosted Veya `veya-serve` predictions. Supports ESM and
CommonJS on Node.js 22+, and browsers with `fetch`, `AbortController`, and
`structuredClone`. TypeScript consumers need TypeScript 5 or newer.

Inference runs in your self-hosted `veya-serve` process. JavaScript clients do
not need Python, PyTorch, or model weights installed; the server does. This
package does not run models directly in a browser or Node.js.

## Run from this repository

In a terminal at the repository root, start the server:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[serve]'
VEYA_HOST=127.0.0.1 VEYA_MODELS=english veya-serve
```

In a second terminal, build the SDK and run the plain JavaScript example:

```sh
cd sdk/typescript
npm ci
npm run build
node examples/triage.mjs
```

The server preloads checkpoints at startup by default. The command above loads
the English checkpoint; the router can load others when needed. To preload both
English and multilingual on CPU:

```sh
VEYA_HOST=127.0.0.1 VEYA_MODELS=english,multilingual VEYA_DEVICE=cpu veya-serve
# Set VEYA_DEVICE=cuda or mps for a suitable GPU installation.
```

Set `VEYA_PRELOAD=0` to defer loading until the first prediction. Initial loads
download weights and need sufficient memory and disk space. The client timeout
defaults to 120 seconds; the example allows 10 minutes for a cold load. Server
configuration uses environment variables, not command-line flags. See the
[server guide](https://github.com/tgundhus/Veya/blob/main/docs/guide.md#http-server-jev-compatible).

## Install in another JavaScript project

Build a tarball from this checkout and install it in your application. Registry publication is a
separate release step; the commands here use the source you have reviewed:

```sh
# From sdk/typescript:
npm pack

# From your app, substitute the actual checkout path:
npm install /path/to/veya/sdk/typescript/veya-client-0.1.0.tgz
```

```js
import { Veya, triageQuestions } from 'veya-client';

const veya = new Veya({ baseURL: 'http://127.0.0.1:8000' });
const result = await veya.predict(
  { message: 'I was charged twice. Please refund the duplicate.' },
  triageQuestions(),
);

console.log(result.answers.intent.choice);
console.log(result.answers.refund_requested.noul); // P(true)
console.log(result.routing?.model); // Veya-only metadata
```

For a group of states sharing one question set, `predictBatch` sends one HTTP request and returns
answers in input order:

```js
const batch = await veya.predictBatch(
  [{ message: 'Refund the duplicate payment' }, { message: 'Cannot sign in' }],
  triageQuestions(),
  { batchSize: 8, sortByLength: true },
);
console.log(batch.results.map(result => result.answers.intent.choice));
```

`batchSize` limits states per forward pass; `sortByLength` can reduce padding. Both are optional.
The client rejects malformed batches before sending them and checks every returned prediction.
Use non-null states. Noul questions accept `labels: { false: 'Reject', true: 'Approve' }`; questions
also accept `option_order`, a permutation of their option indices, while answers keep your own keys.

CommonJS works too:

```js
const { Veya, triageQuestions } = require('veya-client');
const veya = new Veya();
veya.predict({ message: 'Please refund my order' }, triageQuestions())
  .then(result => console.log(result.answers));
```

## Typed custom questions

Question IDs, primitive types, and choice labels are inferred. Use
`defineQuestions` to preserve literal types when storing a schema in a variable.
Inline schemas also infer automatically.

```ts
import { Veya, defineQuestions } from 'veya-client';

const veya = new Veya();
const questions = defineQuestions({
  department: {
    type: 'choice',
    instructions: 'Which department should handle this request?',
    criteria: {
      billing: 'invoices, payments, refunds',
      technical: 'bugs, outages, integrations',
    },
  },
  urgency: {
    type: 'score',
    instructions: 'How urgent is this request?',
    criteria: ['no time pressure', 'soon', 'blocking issue or deadline'],
  },
  refund: { type: 'noul', instructions: 'Does the user ask for a refund?' },
});

const result = await veya.predict({ body: 'Refund the duplicate charge today' }, questions);
result.answers.department.choice; // 'billing' | 'technical'
result.answers.urgency.score;      // number: expected rubric index, from 0 to 2
result.answers.refund.noul;        // number: P(true), from 0 to 1
result.answers.department.probabilities.billing;
result.answers.department.confidence;
result.answers.department.answer_confidence; // number: max(p), the same quantity on every type
result.answers.urgency.abstention;           // 'passed' | 'abstained' | 'unevaluated', with minConfidence
result.answers.urgency.low_confidence;       // true only on an answer that fell below it
result.answers.department.action?.act_probability; // Veya-only metadata
result.usage.input_tokens;
result.usage.truncated;              // boolean: the state did not fit the room left for it
result.usage.truncated_questions;    // string[]: the questions whose own window was cut
```

Choice criteria also accept an array of unique labels, serialized as an option-to-null
map for the shared protocol. Structured JSON values are
supported in instructions and descriptions, as in Python. State may be text, a
JSON object, an array (including chat turns), or null. Undefined values, dates,
non-finite numbers, BigInts and cycles are rejected locally instead of silently
changing during serialization. Empty questions or criteria are rejected.

Response fields retain Python's spelling (`input_tokens`, `act_probability`,
etc.). Scores use zero-based rubric levels. Confidence and accuracy have the
same calibration limits as the Python model; see the root README.

`usage` is the whole record `/v1/systemone` reports: `input_tokens`,
`output_tokens` (always 0), `state_tokens`, `state_tokens_dropped`,
`truncated` and `truncated_questions`. How much of a state the model reads is a
token budget that moves with `max_len`, `head_max_len` and each question's own
option prompt, so these are the only fields that show a cut: the answer itself
is unchanged when evidence is dropped. `options` is added only when some
question's options no longer have a token span each. See the `usage` table in
`docs/http-api.md` for what each key measures.

## Local model selection

By default, `veya-client` sends no `model` property. `veya-serve` then chooses
the appropriate local checkpoint automatically. Set a client-wide `model` or
override it per prediction only when a specific local checkpoint is required:

```js
const veya = new Veya({ model: 'english' });
await veya.predict(state, questions, { model: 'typed-decisions' });
await veya.predict(state, questions, { model: 'multilingual' });
```

Veya accepts its local checkpoint aliases. The HTTP endpoint forwards the
per-request controls documented in the [HTTP API](https://github.com/tgundhus/Veya/blob/main/docs/http-api.md)
reference, and this client exposes them as options on `predict()`:

| option | wire field | meaning |
|---|---|---|
| `model` | `model` | checkpoint name or alias for this request; a path or unpublished Hub id is a 422 from the server |
| `task` | `task` | force a workflow; an unknown name is a 422 naming it |
| `lang` | `lang` | a language code that skips detection when it names a language |
| `langGuess` | `lang_guess` | a code from your own LID, consulted before detection |
| `maxLen` | `max_len` | total token window for this request (server-capped) |
| `headMaxLen` | `head_max_len` | token window the option prompt shares, same cap |
| `minConfidence` | `min_confidence` | abstention gate: one threshold in `[0, 1]`, or a per-bucket map of option-count bucket to threshold; low-confidence answers are marked, not dropped |

```js
await veya.predict(state, questions, { task: 'typed', minConfidence: 0.8 });
await veya.predict(state, questions, { lang: 'de', maxLen: 4096, headMaxLen: 512 });
```

Absent options stay absent, so the deployment's own `Router(...)` settings remain
in charge. The endpoint does not expose a standalone `route()` call. Veya still
detects language automatically; set `VEYA_AUTO_TASK=1` on the server to enable
automatic workflow routing.

Veya returns `model`, `answers`, and `usage`, with optional `routing`, answer
`action`, and Noul `confidence` metadata. Those fields are validated when
present. Choice and Score confidence remains required. Every answer also carries
`answer_confidence` (`max(p)`), the one confidence that means the same thing on all
three question types; when you pass `minConfidence`, each answer of that call additionally
reports `abstention` and `abstention_threshold`, and the ones below the threshold carry
`low_confidence: true`. With `minConfidence` unset none of those three keys appear — absence is
how a caller tells an ungated run from a cleared gate. `health()` is never
called by `predict()`.

## Which JavaScript client?

Use `veya-client` when a JavaScript or TypeScript application talks over HTTP to
self-hosted Python `veya-serve`. Use `veya-ts` when inference must run directly
inside JavaScript through its local ONNX runtime, without a Python server.

## Presets

Available preset functions: `triageQuestions()`, `emailQuestions(categories?)`,
`guardQuestions()`, `moderationQuestions()`, and `routerQuestions()`. Each returns
a fresh schema. Email categories can be customized:

```js
import { emailQuestions } from 'veya-client';
const questions = emailQuestions({ finance: 'payments', engineering: 'bugs' });
```

## Errors, cancellation, authentication

```js
import { Veya, VeyaAPIError, VeyaTimeoutError } from 'veya-client';

const client = new Veya({
  baseURL: 'http://127.0.0.1:8000',
  apiKey: process.env.VEYA_API_KEY,
  timeoutMs: 120_000,
});
const controller = new AbortController();
try {
  await client.predict(state, questions, { signal: controller.signal, timeoutMs: 300_000 });
} catch (error) {
  if (error instanceof VeyaAPIError) console.error(error.status, error.code, error.details);
  else if (error instanceof VeyaTimeoutError) console.error('Request timed out');
  else throw error;
}
```

`timeoutMs: 0` disables the deadline. Calling `controller.abort()` stops waiting;
inference already running on the server may still finish. Requests are never
automatically retried. Other errors extend `VeyaError`:
`VeyaValidationError`, `VeyaConnectionError`, `VeyaAbortError`, and
`VeyaResponseError` (invalid JSON or incompatible response structure).

The client accepts custom `headers` and a Fetch-compatible `fetch` implementation.
Keep API keys on your application's backend. For browser use, put a same-origin
application proxy in front of a protected server, or configure an appropriate
authenticated gateway. A shared server key embedded in browser code is public.

Veya's server binds to `0.0.0.0` by default; the quickstart sets `VEYA_HOST=127.0.0.1`
for local use. Set `VEYA_API_KEY` in its environment to require Bearer
authentication for predictions. `/health` remains public for liveness; `loaded` and `device` need the bearer. `HF_TOKEN`, if needed
for checkpoint downloads, is a separate server-only credential. Configure CORS
and TLS at your application proxy or gateway.

## Python service contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | None for liveness | `{status}`, plus `loaded` and `device` with the bearer |
| `POST /v1/systemone` | `{state, questions, model?}` | `{model, answers, usage}` plus optional `routing` |

Veya errors use FastAPI's `{detail: ...}` body: 400 for malformed requests, 401
for authentication, 413 for request limits, 422 for invalid questions, and 500
for inference failures. The SDK preserves the status and details in
`VeyaAPIError`; string details become the error message. It also accepts
`{error: {code, message, details?}}` from compatible services. OpenAPI is
available at `/openapi.json` and interactive request documentation at `/docs`.

For custom checkpoint paths or an existing agent, configure Python's router:

```python
import uvicorn
from veya import Router
from veya.serve import create_app

router = Router(models={"english": "/path/to/checkpoint"}, device="cpu")
router.preload(["english"])
app = create_app(router)  # Reads VEYA_API_KEY from the environment.
uvicorn.run(app, host="127.0.0.1", port=8000)
```

This SDK uses the repository's existing server without changing its Python
implementation. The integration fixture injects a router into that same server.

## Development

```sh
# In sdk/typescript:
npm ci
npm test
PYTHON=../../.venv/bin/python npm run test:integration
python3 scripts/sync_presets.py --check
```

Integration testing uses real local PyTorch inference with a tiny random test
checkpoint, compares answers against direct Python calls, and downloads no model
weights. It verifies transport and runtime compatibility, not pretrained model
accuracy. Run the repository's pretrained model tests separately when those
weights are available. Presets are generated from `veya/presets.py`; regenerate
with `python3 scripts/sync_presets.py` after editing that source.

## Publishing

Publish `veya-client` from `sdk/typescript` using an npm account with publishing
rights for that package name.

```sh
npm login --registry=https://registry.npmjs.org
npm whoami --registry=https://registry.npmjs.org
npm ci
npm test
PYTHON=../../.venv/bin/python npm run test:integration
python3 scripts/sync_presets.py --check
npm publish --dry-run
npm publish
```

`publishConfig` sets public access and the official npm registry. Publishing runs
the build, type checks, and SDK tests through `prepublishOnly`; `prepack` rebuilds
the distributable. Only `dist/`, this README, the license, and package metadata
are included. Complete any authentication challenge npm presents locally.

The first version is `0.1.0`. For later releases, run
`npm version patch --no-git-tag-version` (or `minor` / `major`) here, then repeat
the checks and publishing commands. Keep SDK versions independent of the Python
package. npm does not allow republishing an already-used name/version pair.

After publication, verify the release and install it into a clean project:

```sh
npm view veya-client version --registry=https://registry.npmjs.org
npm install veya-client
```
