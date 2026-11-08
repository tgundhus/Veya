# Docker quickstart

Run the SDK without installing Python or PyTorch on your host. For the CPU
quickstart, allow 8 GB of RAM and 10 GB of free disk, with Docker Engine or
Docker Desktop and Compose v2 or newer.

From the repository root:

```bash
docker compose run --build --rm veya
```

This builds the checkout, runs the [sample request](https://github.com/tgundhus/Veya/blob/main/examples/docker/request.json)
on CPU and prints JSON covering `choice`, `score` and `noul`. The first request
downloads the selected public Hugging Face checkpoint; no account is needed.
Allow several minutes for its first download.
Weights stay in a named volume. Subsequent runs use `docker compose run --rm veya`.

Predictions and confidence still need evaluation on your workload. See the
[benchmark limits](https://github.com/tgundhus/Veya/blob/main/BENCHMARKS.md).

For ARM64 hosts, DGX Spark and Apple Silicon, see
[ARM64 and DGX Spark containers](docker-platforms.md).

## NVIDIA GPU / CUDA

Install a compatible NVIDIA driver and configure Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The GPU image uses PyTorch CUDA 12.8 wheels. Check your GPU's compute capability
and driver against [PyTorch's supported builds](https://pytorch.org/get-started/locally/);
older cards may require a different build. Allow additional disk space for CUDA
layers. VRAM needs depend on the checkpoint, batch size and input length.

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --build --rm veya
```

The override selects GPU `0` and defaults to `VEYA_DEVICE=cuda`. Set
`VEYA_GPU_ID` to another host index or UUID. That GPU appears as device `0`
inside the container. Check access without downloading weights:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --rm veya python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").cpu())'
```

The sample rejects unavailable CUDA before loading a checkpoint. Veya can still
fall back to CPU after a memory or inference error, so inspect its warnings.
Rebuild when switching between CPU and CUDA configurations.

The image sets `TORCH_DISABLE_NATIVE_JIT=1`. PyTorch 2.14 otherwise replaces some eager
CUDA ops with Triton kernels that it compiles on the first inference, which needs a C
compiler the slim image does not carry: the container reports healthy and then fails every
request (#365). The stock kernels give the same answers at the same latency. Set the same
variable on a bare-metal install if `predict` fails with `Failed to find C compiler`.

This uses [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).
Windows requires Docker Desktop's supported WSL2 GPU setup. Apple MPS,
AMD/ROCm and Intel GPU containers are outside this quickstart; use CPU unless
you configure and validate another backend.

## Configuration

Set Compose variables in your shell, a local `.env` file, or the service's
`environment` block. Don't commit secrets in `.env`. Runtime variables also
work with `docker run -e`; Compose-only settings are identified below.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VEYA_DEVICE` | `cpu` / `cuda` | Device selected by the base / GPU configuration |
| `VEYA_CUDA_AMP` | unset (checkpoint's `amp_dtype`) | `fp16`/`float16` or `bf16`/`bfloat16` for reduced precision, or `fp32`/`float32` to disable autocast. Values are trimmed; an invalid nonempty value raises `ValueError`. Below compute capability 8 either half-precision request runs fp16. The 16-bit TileLang fast path cannot run an fp32 agent. See the [confidence section](guide.md#confidence-and-abstention) for measured probability changes |
| `VEYA_CPU_AMP` | unset | `bf16` or `bfloat16` opts the CPU forward into bf16; anything else leaves it fp32. No fp16 spelling turns it on either: CPU autocast has no fp16 fast path that beats fp32, so bf16 is the only reduced precision core offers on this device |
| `VEYA_MODEL` | `auto` | Router alias: `auto`, `english`, `multilingual`, `typed-decisions` |
| `VEYA_MODEL_PATH` | unset | Compatible checkpoint path inside the container |
| `VEYA_REVISION` | unset | Hub commit, branch or tag used for every checkpoint download, or `reviewed` for the reviewed SHAs in `veya/revisions.py`; a `revision=` argument still wins |
| `VEYA_REQUEST_FILE` | bundled request | JSON request path inside the container |
| `OMP_NUM_THREADS` | `4` | CPU threads; keep within available cores |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset | Optional Hugging Face credential |
| `VEYA_API_KEY` / `VEYA_API_KEY_FILE` | unset | **`veya-serve` only:** require `Authorization: Bearer <key>` |
| `VEYA_PORT` | `8000` | **`veya-serve` only:** container port, and the host port published for it |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints |
| `HF_HOME` | `/home/veya/.cache/huggingface` | Cache path; see mount requirement below |
| `VEYA_CACHE_VOLUME` | project model cache | **Compose only:** named cache volume |
| `VEYA_GPU_ID` | `0` | **Compose only:** NVIDIA device index or UUID |
| `VEYA_TORCH_INDEX` | `cu130` (CUDA) / `cpu` | **Compose build:** PyTorch wheel index. `cu126` also carries the pinned torch; `cu128` does not |
| `VEYA_TORCH_VERSION` | `2.14.0` | **Compose build:** pinned PyTorch version |

Compose forwards the runtime variables except `HF_HOME`, which stays aligned
with its fixed cache mount, and except `VEYA_MPS_AMP_MIN_ROWS`, the MPS row gate,
which no image here can reach because no container here can select MPS.
If overriding `HF_HOME` in `docker run` or your own
Compose file, provide a matching mount writable by UID 10001. Direct Docker
builds select PyTorch with `--build-arg TORCH_INDEX=cu130`; runtime `-e` cannot
change the installed wheel.

```bash
VEYA_MODEL=english OMP_NUM_THREADS=2 docker compose run --build --rm veya

docker build -t veya:local .
docker run --rm -e VEYA_MODEL=english -e OMP_NUM_THREADS=2 \
  -v veya-model-cache:/home/veya/.cache/huggingface veya:local
```

For your own request:

```bash
docker compose run --rm --volume "$PWD/request.json:/inputs/request.json:ro" \
  --env VEYA_REQUEST_FILE=/inputs/request.json veya
```

For a commented configuration with request, checkpoint and secret-file mounts,
see [`compose.example.yml`](https://github.com/tgundhus/Veya/blob/main/compose.example.yml):

```bash
docker compose -f compose.yaml -f compose.example.yml run --build --rm veya
```

Add `-f compose.cuda.yaml` before `run` for NVIDIA GPUs. The example is an
override of `compose.yaml`, so cache and image settings stay in one place.

## Secret files

`HF_TOKEN_FILE` reads a mounted UTF-8 file at startup, trims surrounding
whitespace and takes precedence over `HF_TOKEN`. Unreadable, empty or invalid
files stop startup without printing their contents. The file must be readable
by UID 10001. `_FILE` applies only to supported secrets, not every setting.

With `HF_TOKEN_PATH` pointing to an existing host file outside the checkout:

```bash
docker compose run --rm --volume "$HF_TOKEN_PATH:/run/secrets/hf_token:ro" \
  --env HF_TOKEN_FILE=/run/secrets/hf_token veya
```

Docker secrets or Kubernetes Secret volumes can supply the same file. Values
are loaded into the process environment at startup; restart after changing a
file. Never use tokens as build arguments or bake them into images. Public
checkpoints need no token.

## Fine-tuned checkpoints

This image runs inference. Fine-tuning happens outside it — the
[fine-tuning notebook](https://github.com/tgundhus/Veya/blob/main/notebooks/veya_finetune_typed_decisions_2xT4_kaggle.ipynb)
runs the whole loop on Kaggle's free 2xT4 GPUs and exports a checkpoint this image can
serve. Background and open questions about the training interface stay in
Veya issues #4 and
#26.

Point `VEYA_CHECKPOINT_PATH` to an absolute host directory containing
`rl_agent_config.json`, `model.safetensors` and matching tokenizer files:

```bash
docker compose run --rm --volume "$VEYA_CHECKPOINT_PATH:/models/custom" \
  --env VEYA_MODEL_PATH=/models/custom veya
```

Use a working copy writable by UID 10001 because the loader may update tokenizer
configuration. A LoRA adapter alone is not a complete checkpoint. Leave
`VEYA_MODEL=auto` when setting `VEYA_MODEL_PATH`; an explicit alias and local path
are mutually exclusive. The local-path response comes from the Agent and has no
Router `routing` metadata. These settings also work with the CUDA override.
Evaluate fine-tuned checkpoints on held-out examples before relying on them.

## Models from ModelScope

When a host cannot reach huggingface.co, the checkpoint can come from
[ModelScope](https://modelscope.cn) and be baked into the image at build time. One
argument selects which checkpoint, and it defaults to the multilingual one. The
Compose override adds the prefetch arguments to both services and keeps the
container off the Hub:

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.modelscope.yaml up --build veya-serve
```

For NVIDIA, add `-f compose.cuda.yaml` before `up`; it repeats its own arguments for
both services, so the ordering between the two overrides does not matter. Plain
Docker takes the arguments directly:

```bash
docker build --build-arg MODELSCOPE_MODEL=multilingual \
  -t veya:local .
docker run --rm -e HF_HUB_OFFLINE=1 -p 127.0.0.1:8000:8000 veya:local veya-serve
```

One prerequisite on a host that has run this before. The baked weights land in
`$HF_HOME/hub` inside the image, under the cache directory `compose.yaml` mounts
the `model-cache` volume at (`/home/veya/.cache/huggingface`), and Docker seeds a
named volume from the image only while that volume is empty. A volume left over
from the Hub-based quickstart holds the older Hub snapshot, it is never
re-seeded, and the baked weights stay invisible behind it: the loader resolves
`refs/main` to the old Hub commit and the container answers with the weights
that were already downloaded, as if the rebuild had changed nothing. Point the
deployment at an empty cache volume -- `docker compose down --volumes` with the
same Compose files and the same `VEYA_CACHE_VOLUME`, or `VEYA_CACHE_VOLUME=<name>`
for a fresh one. With `HF_HUB_OFFLINE=1` a missing or divergent ref is a load
failure with no network to fall back to, but the prerequisite is the same.

[`docker/prefetch_modelscope.py`](https://github.com/tgundhus/Veya/blob/main/docker/prefetch_modelscope.py)
lists the repository on modelscope.cn, downloads the checkpoint's own files -- the
same set `veya/agent.py` asks the Hub for, so no sibling checkpoint is pulled -- and
writes them into the image's hub cache the way `snapshot_download` lays out a
snapshot. Nothing else changes: `Agent`, the `Router` that `veya-serve` builds,
`veya.cli` and the integrations keep their repo ids and resolve them to the baked
snapshot, so a container built this way needs no network at all. Each file's size
is checked against what the repository reports before the snapshot is published,
and its SHA-256 as well when the repository publishes one. A size or digest
mismatch fails the build. A repository that publishes no digest leaves the
check to size alone, which cannot detect a same-size substitution.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODELSCOPE_MODEL` | `multilingual` (Compose); empty in the Dockerfile | Checkpoint to bake: `multilingual`, `english`, `typed-decisions` or `all`. Empty means no prefetch and the unchanged image |
| `MODELSCOPE_REVISION` | `master` | ModelScope branch, tag or commit to bake |
| `HF_HUB_OFFLINE` | `1` in the Compose override | `1` never contacts the Hub, so the baked copy is served |

A type expands to that checkpoint's path inside the bundled repository, which is what
the `Router` and the one-shot quickstart load by default, so a build that names one
type serves it with no further change:

```bash
MODELSCOPE_MODEL=english docker compose -f compose.yaml -f compose.http.yaml -f compose.modelscope.yaml up --build veya-serve
MODELSCOPE_MODEL=all docker compose -f compose.yaml -f compose.http.yaml -f compose.modelscope.yaml up --build veya-serve
```

`all` is the whole family, about 2.4 GB of weights. The override also sets
`VEYA_MODELS=multilingual`, because `VEYA_PRELOAD=1` with the default list would try
to build every checkpoint and fail on the first one that was not baked; set
`VEYA_MODELS` to the list you baked when you bake more, and `MODELSCOPE_MODEL=all`
when a deployment really does serve the family.

Several types can be named at once -- `MODELSCOPE_MODEL="english multilingual"`
bakes both, about 1.5 GB -- which is what a serving image usually wants: the
`Router` chooses between the English and the multilingual checkpoint on its own,
and one it was not given answers `500 inference failed` with
`does not contain 'rl_agent_config.json'` in the log. Checkpoints that share a
repository always bake into one snapshot, because a cached revision resolves to a
single directory; the root checkpoint and each subfolder are both in it.

Beyond the types the argument also takes `repo[:subfolder]` specs, comma- or
space-separated, which is how the mirror's standalone repositories
([veya](https://modelscope.cn/models/convaiinnovations/laya),
[laya-multilingual](https://modelscope.cn/models/convaiinnovations/laya-multilingual),
[laya-typed-decisions](https://modelscope.cn/models/convaiinnovations/laya-typed-decisions))
or a fine-tuned checkpoint is baked. A standalone repository is what
`Agent("convaiinnovations/laya-multilingual")` loads directly; the `Router`'s
default is the bundled path, so a type is normally what a serving image wants.

Two details about pins and provenance. The build prints the commit the snapshot is
keyed by, which is the tip of the revision that was baked -- pass that SHA as
`revision=` or `VEYA_REVISION` to pin a load to exactly what was baked. A mirror
repository can hold files from several uploads, so that tip is the one
repository-wide key there is. Hub-side pins do not describe a mirror snapshot:
`reviewed` names Hugging Face commits, and the SHA-256 digest map is keyed to Hub
artifact hashes, so neither applies here, and no digest pin exists for a mirror
snapshot either. The build already refuses a download that does not match the
mirror's own reported size -- and its digest, when the mirror publishes one --
but this checks consistency with mirror metadata, not an independently pinned
digest. The weights come from
whichever mirror account the argument names, which is its own supply-chain
decision for the deployment to make.

## Development and cleanup

Open a Python prompt with `docker compose run --rm veya python`. To run the
existing routing/criteria checks and secret-file tests against your checkout
without downloading weights:

```bash
docker compose run --rm --volume "$PWD:/workspace:ro" --workdir /workspace veya \
  sh -ec 'python tests/test_router.py; python tests/test_criteria.py; python tests/test_docker_entrypoint.py'
```

Rebuild with `--build` after changing source or the bundled example. The image
runs as UID/GID 10001. New named volumes inherit the image cache directory's
ownership; host directories must be writable by that UID. Keep model caches
writable for tokenizer compatibility updates.

`--rm` removes completed containers. `docker compose down` retains the cache.
To **delete downloaded weights**, run `docker compose down --volumes` using the
same Compose files and `VEYA_CACHE_VOLUME` setting. The next request downloads
them again; don't remove a cache shared with another project.

## HTTP serving

The image ships `veya-serve`, so the same build that runs the one-shot quickstart can
serve the Jev-compatible API. `compose.http.yaml` adds it as a second service and leaves
`veya` alone:

```bash
docker compose -f compose.yaml -f compose.http.yaml up --build veya-serve
curl -s localhost:8000/health
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  --data @examples/docker/request.json
```

For NVIDIA, add the CUDA override. It repeats the build args and the device reservation
for `veya-serve`, because `veya-serve` is a separate service and overrides for `veya`
never reach it:

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.cuda.yaml up --build veya-serve
```

`up` keeps the service running in the foreground; `-d` detaches. Weights go to the same
named `model-cache` volume as the quickstart, so serving after a quickstart run starts
with the checkpoints already on disk. Stop with `docker compose ... down`, using the same
Compose files.

The port is published on `127.0.0.1` only. The API has no authentication until
`VEYA_API_KEY` is set, so set a key before exposing it with
`VEYA_BIND_ADDRESS=0.0.0.0`, and put a TLS reverse proxy in front for remote clients.
`/health` does not require authentication in either case, so the healthcheck below keeps
working; with a key set it answers an unauthenticated caller `{"status": "ok"}` and withholds the
checkpoint, revision and device fields, which need the bearer.

The service has a healthcheck on `/health`. The server preloads before it starts
listening, so with `VEYA_PRELOAD=1` a healthy container has its checkpoints loaded.
`docker compose ... up -d --wait veya-serve` returns once it is healthy.

`/health` reports `device` as the device a resident checkpoint actually computes
on, which is not always what `VEYA_DEVICE` asked for: a checkpoint that wants a
GPU it cannot get falls back to CPU silently and still answers correctly.
`checkpoint_devices` names each loaded checkpoint, and `device_is_preference` is
`true` only while nothing is resident, so a deployment that quietly lost its GPU
says so instead of echoing its own configuration back.

### Server configuration

These apply to the `veya-serve` service only.

| variable | default | effect |
|---|---|---|
| `VEYA_HOST` | `0.0.0.0` | bind address inside the container |
| `VEYA_PORT` | `8000` | container port, and the host port published for it |
| `VEYA_BIND_ADDRESS` | `127.0.0.1` | host address the port is published on |
| `VEYA_PRELOAD` | `0` | `1` builds every checkpoint at startup instead of on first request |
| `VEYA_MODELS` | (all) | comma list to preload: `english,multilingual,typed-decisions` |
| `VEYA_THREADS` | `OMP_NUM_THREADS` | caps torch intra-op threads; keep at or below physical cores |
| `VEYA_PERFORMANCE_MODE` | unset | Optional [resource preset](performance-modes.md), `1`–`5`, or `default` for current behavior. The example's explicit thread and preload settings override the preset |
| `VEYA_AUTO_TASK` | `0` | `1` lets the router reach `typed-decisions` automatically |
| `VEYA_DEFAULT_MODEL` | `english` | Checkpoint a state with no language evidence falls back to (no letters, or Latin text too short to identify). Set `multilingual` for mostly non-English traffic; an unresolvable name stops the container at startup instead of serving a configuration nobody asked for |
| `VEYA_MAX_LOADED` | `2` | Checkpoints kept resident; `VEYA_AUTO_TASK` makes a third reachable on demand, and a cap below what routing chooses rebuilds one per switch |
| `VEYA_MAX_CONCURRENT` | `16` | requests admitted at once; later ones get `503` (a value that does not parse, or is not positive, falls back to `16`) |
| `VEYA_LOG_LEVEL` | `info` | uvicorn log level |
| `VEYA_API_KEY` | (none) | when set, requires `Authorization: Bearer <key>` |
| `VEYA_ROOT_PATH` | (empty) | public URL prefix for FastAPI when behind a reverse proxy; the proxy should strip it before forwarding |
| `VEYA_MAX_TOKEN_BUDGET` | `8192` | cap on per-request `max_len` and `head_max_len` overrides |
| `VEYA_MAX_BATCH_TOKENS` | `131072` | tokens one batch forward pass may collate; larger batches are split into several passes |
| `VEYA_MICROBATCH_MAX_STATES` | `0` | Opt-in collection of compatible single HTTP requests; 0 disables, otherwise 2–64 |
| `VEYA_MICROBATCH_WAIT_MS` | `1` | Finite nonnegative collection window in milliseconds; see [microbatching](microbatch.md) |
| `VEYA_BODY_READ_TIMEOUT` | `10.0` | seconds allowed to receive a request body; a stalled upload gets `408` and releases admission |
| `VEYA_IDLE_UNLOAD_SECONDS` | `0` | unload idle checkpoints after this many seconds; the next request loads them again |
| `VEYA_JEV_STRICT` | `0` | `1` projects responses onto the strict Jev wire contract, removing Veya extensions |
| `VEYA_TOURNAMENT_GROUP` | `0` | Opt-in HTTP choice elimination in groups of 2–100; choices can then contain up to 255 labels. Ordered choices keep the one-pass cap |
| `VEYA_SHA256_DIGESTS` | (none) | JSON digests checked before a checkpoint is parsed: `{artifact: digest}` for every checkpoint, or `{model: {artifact: digest}}` per checkpoint. See [Security](security.md) |
| `VEYA_EXTRA_MODELS` | (none) | Additional checkpoint names and paths as JSON |
| `VEYA_CACHE` | `0` | Enable process-local memory decision caching |
| `VEYA_CACHE_PATH` | (none) | SQLite decision file; a nonempty path enables caching |
| `VEYA_CACHE_TTL` | `86400` | Fixed retention in seconds, or `none` |
| `VEYA_CACHE_MAXSIZE` | `10000` | Decision count target, or `none` for no capacity eviction; SQLite enforces numerical bounds during periodic pruning |
| `VEYA_CACHE_FINGERPRINT` | (none) | Explicit policy identity across model upgrades; empty uses automatic model/configuration identity |
| `VEYA_CACHE_REQUIRED` | `0` | Require successful replay or commit; storage failure returns 503 |
| `VEYA_CACHE_REQUIRE_EXISTING` | `0` | Refuse to create a missing SQLite decision database |
| `VEYA_CACHE_EXPECTED_STORE_ID` | (none) | Require an existing database with this store identity |
| `VEYA_REPLAY_MANIFEST` | (none) | Path inside the container to verified local replay metadata |
| `VEYA_DECISION_POLICY` | (empty) | Override the checkpoint default with `canonical` or `context-balanced-v1` |
| `VEYA_HTTP_REPLAY_FAST_PATH` | `1` | Let verified hits bypass inference queues when a manifest is configured |
| `VEYA_CACHE_RENEW_ON_HIT` | `0` | Extend retention on replay |

For example, set `VEYA_ROOT_PATH=/veya` when publishing the API under `/veya`. The proxy must
strip that prefix before forwarding to the container; this setting updates FastAPI's generated
URLs and does not change the internal `/health` or `/v1/systemone` routes.

`VEYA_PRELOAD` defaults to `0` here rather than the package default of `1`, because
preloading makes the first boot download all three checkpoints. Set it to `1` for a
long-running deployment so the first request does not pay for the build.

`VEYA_PORT` sets both the published host port and the port the server binds, so the two
cannot drift. Change one place to move the service:

```bash
VEYA_PORT=9000 docker compose -f compose.yaml -f compose.http.yaml up --build veya-serve
```

For a persistent seven-day decision cache, use the container's dedicated volume path:

```bash
VEYA_CACHE_PATH=/home/veya/decisions/cache.sqlite VEYA_CACHE_TTL=604800 \
  docker compose -f compose.yaml -f compose.http.yaml up --build veya-serve
```

The `decision-cache` volume is separate from downloaded models. Removing that volume removes
its database files; TTL alone does not physically erase rows or backups. See
[retention and deletion](production.md#retention-and-deletion).

For decisions retained without expiry or capacity eviction, set `VEYA_CACHE_TTL=none`,
`VEYA_CACHE_MAXSIZE=none` and `VEYA_CACHE_FINGERPRINT=policy-v1` with that persistent
SQLite path. Increment the policy identity when earlier decisions should be
reevaluated. Add `VEYA_CACHE_REQUIRED=1` to refuse successful predictions on storage
failure; optional mode permits recomputation. Removing the volume loses its decisions;
see [the durable replay limits](production.md#cache-configuration).

### Bearer token from a file

`VEYA_API_KEY_FILE` is read once at startup, moved into `VEYA_API_KEY`, and the `_FILE`
variable is removed before the server execs. Prefer this to putting the key in the
environment:

```bash
docker compose -f compose.yaml -f compose.http.yaml run --rm \
  --volume "$PWD/veya_api_key:/run/secrets/veya_api_key:ro" \
  -e VEYA_API_KEY_FILE=/run/secrets/veya_api_key \
  --service-ports veya-serve
```

