# Upstream Laya releases adopted by Veya

The original project's release notes for the versions merged into Veya, from its README. Veya
0.3.29 contains everything below, except the Java and .NET SDKs, which it does not carry, and with
Veya's own additions on top: see the [README](https://github.com/tgundhus/Veya#readme). Issue
and pull request numbers refer to the original project.

## What's new in 0.3.29

* **`predict_long` no longer splits a state that fits.** The one-pass shortcut compared the state against the default window instead of the room the questions actually leave, so any state in the band between the two was scanned in two windows and answered with the max or the most confident window rather than the plain answer (#966). On `laya-multilingual` that band is roughly 761 to 989 state tokens, an ordinary email: an 878-token note returned noul 0.998 against `predict`'s 0.752, at twice the forward passes. It now matches `predict` exactly, and an explicit `window=` still scans.
* **A line of English acronyms no longer routes to the multilingual checkpoint.** `MON`, `DES`, `EST` and `LA` collide with French stopwords, so one line of ordinary abbreviations inside an English state made `analyse` name the whole state French (#1013). The all-caps guard is kept, because a customer shouting in Portuguese is still Portuguese, and the fix separates that case from a line of bare acronyms. A second cause is fixed with it: acronyms were voting in the whole-state verdict too, which no helper was filtering.
* **A `min_confidence` map key that cannot name a bucket is refused.** The validator checked only that a key was a string, so `{"choice:2-5": 0.9}` passed and then silently gated nothing: the lookup misses, falls back to `default`, then to 0.0 (#1002). A caller who mis-spells one key believed that bucket was gated at 0.9 while every answer in it abstained never.
* **Fine-tuning reports what it learned, on held-out data.** `veya-train --eval eval.csv` evaluates the base checkpoint before training and the fine-tune after, and writes `train_report.json` with accuracy, loss, ECE, Brier and signed deltas (#967, #887). Evaluation items are checked for overlap against both the training and calibration splits, and a set that overlaps is labelled `overlapping_eval` rather than reported as generalization. With no `--eval` and no calibration items, evaluation is skipped rather than reporting training fit.
* **A fine-tune that collapses to the class prior now says so.** On a few hundred rows the default epoch budget can finish at chance with a near-constant logit, and the run still looks successful (#968, #963).
* **`veya-evals evidence --checkpoint DIR`** reads the calibration evidence a fine-tune persists and reports it as `PRESENT`, `MISSING`, `INSUFFICIENT` or `UNKNOWN`, so zero calibration items stops reading as healthy (#964). It loads no weights and does not need torch.
* **The fine-tuning scripts now share one loop.** `notebooks/veya_finetune_typed_decisions_mps.py` and `research/scripts/finetune_single_device.py` call `veya.train` instead of carrying their own copies, which is what #851 and #885 were: the same temperature-clamp bug fixed twice in parallel (#965, #887).
* **The default loss stays `rlcd`, now on evidence.** A three-seed panel on typed-decisions says the single-seed soft-CE win does not replicate: paired `soft-ce − rlcd` accuracy is +0.0002 on average and changes sign between seeds, with soft-CE far noisier (std 0.0155 against 0.0016) (#1012, #887, #741).
* **Java.** `veya-java` gains language routing, the preset question sets, the embedding shortlist and the email cleaner, gated against fixtures recorded from the Python package, plus a JDK matrix lane because `\p{L}` follows the JDK's own Unicode version (#938). Three defects on `main` are fixed with it, including a release workflow that could publish a jar named after a branch, and a release test floor that counted skipped tests.
* **Thirty-one documentation corrections**, each landing a gate that fails if the page drifts again (#972 to #1011, from @aashish254). The largest class: `confidence` was documented as one entropy formula across every question type, when it is `1 - H/log(k)` for `choice` and `score` and equals `answer_confidence` for `noul`. Others name parameters a tool actually takes, correct key counts, and stop calling the shipped `answer_confidence` calibrated when the loader itself warns that it is not.
* **Portability.** Six repo reads in `tests/` left the codec to the runner, which is cp1252 on the Windows lane, so a UTF-8 page killed the whole suite at the read rather than failing one assertion. A gate now refuses an unpinned repo read (#1001).
* **`veya[onnx]` no longer caps NumPy for everyone.** The cap that torch below 2.3 needs is scoped by marker to macOS on Intel, instead of making the extra unresolvable next to anything requiring `numpy>=2` (#947).

44 pull requests from 8 contributors.


## What's new in 0.3.28

* **`veya.backends` was missing from every published wheel.** The package was never added to setuptools' explicit list, so the wheels for 0.3.25, 0.3.26 and 0.3.27 shipped without it (#940). The documented backend selection API was therefore unavailable to anyone who installed with pip: `Agent(backend="eager")` raised `ModuleNotFoundError` and `agent.set_backend("eager")` raised `ImportError`, while the compile guide described both. The default path was unaffected, which is why this went unnoticed; it also only ever worked from a source checkout, which every contributor here uses. A packaging check now keeps the declared list aligned with every importable `veya` source package, so the next one fails at build time instead of at install time.
* **Fine-tuning from the command line.** `veya-train --data tickets.csv --out ./ft` fine-tunes a checkpoint from a plain labelled CSV, discovering the labels and synthesising the question (#931, #887). It also reads the `{state, questions, expected}` rows `veya-evals` already takes, so an evaluation set trains without conversion, and `--dry-run` prints the item and skip counts before anything loads. The generated question schema is saved beside the checkpoint, so the labels and wording can be reused exactly at inference.
* **A fine-tune now says how much its calibration can be trusted.** A short run could fit temperatures of `[1.0, 1.0, 1.0]` from a handful of items and save a checkpoint whose confidences were never calibrated, indistinguishable from one that was. `finetune` warns per question type and records the same report in the checkpoint config (#933). Questions whose options run past `max_len` are skipped and counted rather than crashing the batch mid-run (#934), and a final partial accumulation window is normalised by the micro-batches it actually holds (#941).
* **Many-option choices.** `veya.predict_tournament(agent, state, questions)` answers a choice question whose labels do not fit the option budget by splitting them into groups of 16, answering every group in one forward pass and running the winners in a final call (#950). On full test splits: BANKING77 0.430 to 0.610, CLINC150 0.625 to 0.876, with ECE falling from 0.350 to 0.082. No embedder and no new dependency.
* **Order-invariant checkpoints.** An opt-in parallel option layout gives every option the same position ids and blocks attention between options, so reordering can only permute the logits (#951). A checkpoint trained with `TrainConfig(option_layout="parallel")` gives the same answer under reordering for 100% of decisions, against 91.7% for the current layout. Every published checkpoint stays on the default sequential layout and nothing changes for them.
* **Serve your own checkpoints beside the built-ins.** `Router(models=...)` and `Router.register(name, source)` accept any name, with a Hub repo id, a `(repo, subfolder)` pair or a local directory as the source, and the router loads, evicts and unloads it exactly like a built-in (#937, #919). `Router.registered` lists them and `Router.unregister` removes one.
* **pydantic `Enum` fields work.** `questions_from_pydantic`, `decide(schema=Model)` and `answer_to_pydantic` raised `SchemaError` on every enum field, because pydantic v2 renders them as a `$ref` and the planner never resolved one. Local `$defs` references now resolve, including inside `Optional[Enum]` and pydantic v1's one-item `allOf` wrapper (#960).
* **Abstention cuts are fitted on the precision the gate reads.** `fit_abstention_thresholds` chose a cut from full-precision confidences while the gate compares the 4-decimal `answer_confidence`. With a binning map installed the whole cut bin was lost: the fitter judged 532 accepted answers and the gate delivered 382 (#957).
* **`veya-evals` refuses a NaN threshold.** `--min-accuracy nan` and `--max-ece nan` were accepted and every comparison against NaN is false, so a run at 0% accuracy and ECE 0.99 exited 0 (#954).
* **MCP keeps `option_order`.** `veya_predict`, `veya_route`, `veya_shortlist` and every `veya_predict_batch` item dropped the key, so the README's rotation-averaging recipe returned k identical answers (#958).
* **The ONNX agent catches an in-place question rewrite.** A start hook that widens a question in place mutated the before-image `predict_long` compared against, so the scan-budget guard returned early and up to 59% of a document reached no model while `usage` reported every window as read. The torch agent already raised on the identical input (#959).
* **torch 2.0 to 2.3 can run the package again.** The action-head dtype check called a form of `torch.is_autocast_enabled` that only exists from torch 2.4, so every forward pass raised `TypeError` across the bottom of the declared `torch>=2.0.0` range (#945). Two test suites that read `torch.fx.experimental._config` at module level no longer fail collection on a torch without it (#946), and the Windows lane's bf16 legs are skipped on a Windows CPU, where they hit an uncatchable SIGILL (#949).
* **macOS and Intel.** `setup_veya.sh` and `LOCAL_SETUP.md` pin the stack that actually runs there, since a plain `git clone` resolves torch 2.2.2 against a transformers that wants 2.4 and nothing in the error message says so (#948).
* **Docker.** The default base image moves to Debian 13 (trixie), with bookworm one build arg away (#953, #739). Verified with a decision diff across both bases: 291 answers per checkpoint on torch CPU, torch CUDA and ONNX Runtime CPU, zero differing decisions.
* **TypeScript.** `Router.predictBatch` and `predictMany` forward per-call `onPredictStart`, `onPredictEnd` and `hooksRaise`, which were accepted and silently dropped (#936, #935). French mail is cleaned the way Python cleans it, over 4,000 generated messages with zero mismatches (#939). An English word that doubles as a foreign function word is counted once, so plain English like `Come one, come all` is no longer routed to the multilingual checkpoint (#955).
* **.NET and Java.** The French device footer is matched with the spacing people actually write (#956), and an empty action-logit row gives `[0.5, 0.5]` instead of `[NaN, NaN]` (#942).
* **Security.** The Gradle wrapper jar is validated against Gradle's published checksums before anything executes it, and the distribution it downloads is pinned by sha256. The email examples no longer use a real bank's name with two unregistered lookalike domains as the phishing lure; the preset still flags the generic version at 0.926. A full audit of the repository, the published wheel, both npm packages and all ten workflows found no malicious content.

28 pull requests from 12 contributors.


## What's new in 0.3.27

* **A plain `import veya` no longer crashes when TensorFlow is installed.** transformers 4.x imports TensorFlow while veya builds the model, and a broken TF build turns that into `Fatal Python error: Bus error` from a library veya never uses (#915). The package now sets `USE_TF=0` the way CI, the Dockerfile and the examples already did.
* **Per-call hooks reach the batch path.** `Router.predict_batch` takes `hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise` and `hooks_timeout`, matching `predict`; `Router.route_batch` takes `hooks`, `hooks_raise` and `hooks_timeout`, since it fires only `on_route` and has no predict events for the convenience callables to bind to (#909). The batched dispatch also composes installed hooks properly, so a `set_default_hooks` default no longer fires on `predict` and silently vanishes on `predict_batch`.
* **A stale abstention flag is cleared on re-evaluation.** Re-running the gate on a reused result dict with a more permissive threshold used to leave `low_confidence: True` behind, so downstream callers treated the answer as permanently abstained (#910).
* **`predict_long` runs its scan after the hook chain** rather than as a start hook, so veya's own forward pass no longer executes inside `dispatch()` and inherits the caller's hook machinery.
* **Abstention thresholds are fitted on the scale the gate reads.** With a histogram-binning map installed, the runtime reports a binned confidence; `fit_abstention_thresholds` now fits against that instead of the temperature-scaled value, so a fitted cut and the gate agree.
* **Selective-classification metrics no longer split tied confidences.** `selective_accuracy` and `aurc` cut at a row index, so with ties the result depended on dataset order. Ties are the normal case for `answer_confidence`, not a corner.
* **Java.** `veya-java/` is a JVM inference package with the same live parity gate the .NET port uses: fixtures generated from the Python package by a committed script, a drift check, and a skip guard so a missing artifact fails rather than passes. The lane is advisory for the same reason the .NET one is.
* **French email.** Device footers with ordinary spacing are matched, not only the tight spacing the original samples happened to have.
* **MPS autocast, measured on an M1 Pro.** `mps_amp_min_rows` has defaulted to 5 since #109 on the strength of one M5 measurement, hedged for M1 to M5. `BENCHMARKS.md` now carries paired fp32-against-fp16 numbers on an M1 Pro, with the script that produces them.

9 pull requests from 7 contributors.


## What's new in 0.3.26

A small release: one new SDK, one TypeScript port, and the dependency bumps.

* **.NET SDK.** `veya-dotnet/` is a C# port of the inference path running the official checkpoints exported to ONNX on ONNX Runtime, so calling Veya from .NET no longer needs a Python sidecar (#673). It ships with a parity lane that re-records its golden fixtures from the Python package at every commit, so drift surfaces on the pull request that causes it; the lane is advisory, because a Python improvement should not wait on a C# port. Long prediction, per-language temperatures, hooks, batching, option ordering, digest controls and usage diagnostics are not ported yet, and there is no NuGet package.
* **Histogram binning in veya-ts.** `applyBinningMap` / `checkBinningMap` port the non-parametric recalibration from core, including the language-override bypass, and the recalibrated `answer_confidence` is what the abstention gate reads.
* **Dependencies.** `@types/node` 24 to 26, zensical 0.0.65 to 0.0.67, ruff 0.16.8 to 0.16.9, and the CodeQL actions to 4.38.2. The three that could have broken a lane were each verified against that lane before merging.

7 pull requests: 2 from contributors and 5 dependency bumps.


## What's new in 0.3.25

* **Memory on demand.** `VEYA_IDLE_UNLOAD_SECONDS` lets `veya-serve` give its checkpoints back after an idle window, and `VEYA_BASE_URL` lets `veya-mcp-server` answer from a running server over HTTP instead of loading its own copy, so several editor sessions share one resident model (#888). Both default to off.
* **Backends.** `Agent(backend=...)` and `agent.set_backend(...)` select `auto` / `eager` / `compile` / `tilelang` / `onnx` through one class layer, with ONNX routed to the existing `ONNXAgent` rather than a second copy of that path.
* **Compile defaults.** `compile=True` warms during load instead of paying 50 s on the first request, with `compile_warmup=False` to restore the lazy shape. `compile_cache` gives Inductor a Veya-specific directory and `compile_mode` reaches `reduce-overhead`.
* **AOTInductor packaging works.** `DecisionModel.forward` cast to fp32 before the action head while the encoder ran under AMP, which packaged a graph that died with `mat1 and mat2 must have the same dtype`. The cast now matches the head's weight dtype outside autocast only, so eager and AMP numerics are untouched.
* **Oversized batches split instead of collating whole.** `/v1/systemone/batch` chooses a state chunk size when the token product would exceed the cap, rather than building one 4,096-row forward pass. No request that worked before is refused and none changes its answer.
* **Calibration.** Histogram binning is wired into answers and the calibration payload, so a fitted `binning_map` ships and loads like the temperatures.
* **TileLang.** All five kernels now lower for the CPU target as an explicit fp32 specialization, and GEGLU uses two pipeline stages.
* **Stricter device handling.** A CUDA ordinal past `device_count`, and `auto` in any casing, are resolved at load with the same warn-and-CPU shape as the existing fallbacks instead of dying later in `.to()`.
* **Encoding.** `veya` writes redirected stdout as utf-8, the output half of the stdin fix in #799.
* **TypeScript.** Both clients accept the per-bucket `minConfidence` map, and `veya-ts` ports it to the on-device engine.

13 pull requests from 9 contributors.


## What's new in 0.3.24

* **Concurrency.** `Router.load` builds a checkpoint outside the lifecycle lock, so a cold build no longer blocks calls for a checkpoint that is already resident, and `unload` is synchronized per checkpoint rather than globally (#848). `predict_long` holds the tokenizer lock like every other encode, so scanning a document next to ordinary predictions no longer raises `Already borrowed` (#825).
* **Calibration.** Per-option-count abstention thresholds, because one `min_confidence` does not transfer across option counts (#394): `fit_abstention_thresholds` fits one cut per bucket, keyed like `temperature_by_options`, and fails closed on a bucket too unreliable to accept anything. Histogram binning (`fit_binning_map`, `apply_binning_map`) recalibrates a bucket whose reliability curve temperature scaling cannot reach. `save_calibration` writes atomically, and `records_from_labeled` works on a loaded agent (#826).
* **Evals.** Selective-classification metrics (Brier, AURC, selective accuracy) from the same pairs ECE already uses, so the abstention gate has a yardstick. A NaN metric, limit or tolerance can no longer pass a gate (#830).
* **Stricter inputs.** A non-finite `hooks_timeout` (#828), a non-scalar `choice` label, a batch item's untyped `lang_guess`, and an unvalidated `min_confidence` or `hooks_timeout` on `veya_predict_batch` are all refused where they are read. `hooks_installed` accepts a sequence, as documented (#829). A `null` score level is a 422 instead of a response no Jev client can parse.
* **veya-serve.** `VEYA_JEV_STRICT` projects the response onto the strict Jev wire contract for clients that reject unknown fields. `/v1/systemone/batch` sums `output_tokens` instead of reporting zero, and refuses an unpaired surrogate like the single endpoint.
* **Fallback notes are warnings, not prints.** Eight device- and inference-fallback notes went to stdout, which corrupted `veya --json` output; they are `warnings.warn` now, so they are filterable.
* **One test list.** CI and the release gate run the same shared suite list (`scripts/test_suites.py`). They had drifted to 80 suites against 47, so a regression in any of the 33 CI-only suites could reach PyPI.
* **TypeScript.** The SDK types and validates the whole `usage` report, the four confidence fields, and the routing detection's `mixed_segment`. `veya-ts` reports collapsed options like `veya.common` and keeps bytecode out of the npm tarball.
* **Docs and examples.** Nine example and page corrections where the prose contradicted the code, each with a test that holds the page to the source. The PyPI README links absolutely, so its 31 relative links no longer 404 on the project page.

56 pull requests from 16 contributors.


## What's new in 0.3.23

* **Security.** `GET /health` no longer answers deployment internals to an unauthenticated caller on a server that set `VEYA_API_KEY` (#812): liveness stays open so every shipped probe keeps working, while the resident checkpoint names, revision SHAs, device state and fallback reasons need the bearer. A deployment with no key set is unchanged. `SECURITY.md` now documents private vulnerability reporting.
* **Concurrency.** A GPU OOM fallback no longer moves the shared model under another in-flight request (#649). Normal forwards stay concurrent through a reader-writer gate; only the device demotion is exclusive.
* **Per-call controls reach every surface.** `lang`, `lang_guess`, `min_confidence`, `task` and the token budgets now forward through the CLI (`--lang-guess`, `--min-confidence`), `/v1/systemone/batch`, the MCP single-request and batch tools, the TypeScript SDK and the LangChain, CrewAI and LlamaIndex wrappers. The three framework wrappers also gate `confidence_threshold` on `answer_confidence` rather than the entropy confidence, matching core's own gate.
* **Routing.** Swedish is detected with MASSIVE evidence across all 51 locales and no locale regressing. `VEYA_DEFAULT_MODEL` makes the routing fallback settable from the environment, and a per-checkpoint pin no longer silently disables the caller's digest or revision.
* **ONNX.** `--quantize` defaults to per-tensor, because per-channel collapsed the decision model to 32 percent agreement with eager. Export declares its dynamic dims in a spelling every torch in the supported range accepts, and verifies the symbolic axes it wrote.
* **Long documents.** `predict_long` sizes its windows from the room a question actually leaves for the state, so a scan no longer skips part of the document and reports that it read it.
* **Evals.** Opt-in per-slice quality gates catch a slice regressing while the overall metric improves, shortlist runs attribute errors to retrieval or decision, `--min-confidence` reaches the abstention gate, and `--calibration` works on the ONNX path.
* **Abstention reporting.** With `min_confidence` set, every answer reports `abstention` and `abstention_threshold`. With no threshold the payload is byte for byte what it was.
* **Research.** A Spanish phone-turn benchmark (217 frozen sentences, label policy fixed before any model ran) and a Chinese reliability evaluation with source-group isolation and paired bootstrap intervals.
* **Email.** French mail clients are cleaned the way English, Portuguese and Spanish already were.

64 pull requests from 21 contributors. #742 landed inside #684, which had been rebased onto it. Full list in the 0.3.23 release.


## What's new in 0.3.21

* **ONNX catches up with PyTorch.** `ONNXAgent` gains `predict_batch` (with `sort_by_length`), `predict_long` and `decide_batch`, `scripts/export_onnx.py --quantize` writes a per-channel INT8 copy for CPU, and `veya-evals run --onnx` scores an export with the same gates as the torch path.
* **Opt-in abstention.** `min_confidence=` on `predict`, `predict_batch`, `decide` and `decide_batch` flags answers below a threshold on `answer_confidence` with `low_confidence: True`, and `decide` returns `None` for them. When a threshold is set, every answer also reports `abstention` — `passed`, `abstained` or `unevaluated` — so a caller can tell a gate that cleared from a gate that never ran; with no threshold, nothing is added at all.
* **Batch everywhere.** `decide_batch`, `Router.predict_long`, `veya --batch FILE`, the MCP `veya_predict_batch` / `veya_route_batch` / `veya_decide` tools, and LangChain `batch()` / `abatch()` all run on shared forward passes. New `VeyaDecision` (LangChain), LlamaIndex selectors (`veya[llamaindex]`) and CrewAI routing (`veya[crewai]`).
* **Per-request token budget.** `max_len` / `head_max_len` now reach every surface: `veya-serve` (capped by `VEYA_MAX_TOKEN_BUDGET`), `Router.predict_batch` requests, the CLI (`--questions`, `--max-len`, `--head-max-len`), MCP tools and LangChain nodes.
* **Operations.** `VEYA_MAX_LOADED`, `VEYA_REVISION` and per-checkpoint SHA-256 maps; `/health` reports the device a checkpoint really runs on and its CPU-fallback count; the 503 busy answer carries `Retry-After`; `compile=True` no longer recompiles for every request shape.
* **Stricter inputs.** A null or duplicate `choice` label, a short temperature list, a `None` state and non-dict questions are refused with a message that names them, and `usage["options"]` says when the head budget left two options with the same tokens.


<p align="center">
  <img src="https://raw.githubusercontent.com/tgundhus/Veya/main/assets/laya_vs_jev_full.png" alt="Veya versus TypeSafe Jev: accuracy on shared public datasets, every application workflow, all 51 languages, speed, calibration, and the cost of not preloading" width="100%" />
</p>

Veya evaluates typed questions (`choice`, `score`, `noul`) over text, email, tickets or JSON
documents in a forward pass. Archived T4 measurements were 33 ms for one question and
7.2 ms/question batched. It returns typed decisions without generating text; predictions
can still be wrong.

Three checkpoints, and a `Router` that picks between them per request:

| | encoder | params | context | use it for |
|---|---|---|---|---|
| [`veya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English |
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 (up to 8,192) | 100+ languages, 2x faster |
| [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |
