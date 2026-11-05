# Security Policy

Veya, like the Laya engine it is built on, runs in the user's own process or on self-hosted
infrastructure, so its security centres on supply-chain integrity, runtime resource containment,
safe deserialization and network isolation.

---

## Supported Versions

Veya is maintained on its `main` branch, which currently includes Laya 0.3.29. Security fixes
land on `main`; install from it as the README shows.

---

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Veya, `laya-serve`, the TypeScript
packages or the Docker image built from this repository, **please do not disclose it in a public
issue, pull request or discussion.** Report it privately to the maintainer, Tobias Gundhus
([@tgundhus](https://github.com/tgundhus) on GitHub).

A vulnerability in code Veya has not changed (the models, the training method or the engine)
affects the original Laya as well, so it is worth reporting to the original project's maintainers
too.

### What to Include

- **Vulnerability classification**: for example authentication bypass, denial of service, memory
  exhaustion, an unhandled exception, path traversal.
- **Affected component**: the `laya` Python package, `laya.serve`, the decision cache, the MCP
  integration, the TypeScript packages or the Docker image.
- **Affected version**: the commit SHA.
- **Reproduction steps**: a minimal script, curl command or payload.
- **Impact assessment**: severity, prerequisites and a realistic exploit scenario.
- **Remediation** (optional): a suggested fix or mitigation.

---

## Security Architecture & Operational Hardening

When deploying Laya in production environments (especially when self-hosting `laya-serve` or running on shared hosts), adhere to the following security guidelines:

### 1. Model Weights & Supply Chain Integrity
- **Safe Weight Deserialization**: Laya checkpoints rely on modern, safe serialization formats (`safetensors` and ONNX). The engine avoids unsafe arbitrary object pickling.
- **Revision Pinning**: Pass `revision=PINNED_REVISIONS[...]` when loading weights to guarantee that the runtime only downloads and executes reviewed commit revisions from the Hugging Face Hub, protecting against moving branch attacks.
- **Cryptographic Digest Verification**: Set `LAYA_SHA256_DIGESTS` in the environment or pass `expected_sha256` to loaders to enforce SHA-256 fingerprint verification of every artifact prior to parsing and weight loading. See [docs/security.md](docs/security.md) for detailed instructions.

### 2. HTTP Server Hardening (`laya-serve`)
- **Bearer Token Authentication**:
  - `laya.serve` reads the `LAYA_API_KEY` environment variable. When set, all requests to `/v1/systemone` require `Authorization: Bearer <token>`.
  - Authentication comparisons use constant-time `hmac.compare_digest` across UTF-8 encoded bytes with surrogate-escape handling to prevent both timing attacks and latin-1 decoding exceptions.
  - **Warning**: If `LAYA_API_KEY` is unset, authentication is disabled and the server answers requests without credentials. Always set `LAYA_API_KEY` in non-isolated environments.
- **Network Interface Binding**:
  - By default, `laya-serve` binds to `0.0.0.0:8000` (`LAYA_HOST=0.0.0.0`).
  - For local or loopback-only deployments, explicitly set `LAYA_HOST=127.0.0.1` to prevent exposure on external network interfaces.
  - For public or multi-tenant deployments, place `laya-serve` behind a hardened reverse proxy (e.g., NGINX, Envoy, Caddy) that enforces TLS termination, strict rate limiting, connection limits, and request timeouts.
- **Denial of Service & Memory Amplification Protections**:
  - **Option Amplification Guard**: `MAX_CHOICE_OPTIONS = 100`. The server automatically rejects choice questions with more than 100 options with HTTP 413 before running inference, preventing memory and forward-pass amplification attacks.
  - **Token Budget Ceiling**: Per-request overrides (`max_len`, `head_max_len`) are validated and capped by `DEFAULT_MAX_TOKEN_BUDGET = 8192` (configurable via `LAYA_MAX_TOKEN_BUDGET`).
  - **Admission Semaphores**: Concurrent requests buffering large request bodies are bounded by `DEFAULT_MAX_CONCURRENT = 16` (configurable via `LAYA_MAX_CONCURRENT`) to prevent memory exhaustion from slow-body attacks.

### 3. Container & Host Isolation
- **Non-Root Execution**: The official Docker image executes under an unprivileged user `laya` (`UID 10001`, `GID 10001`). Do not override this with `USER root` in production.
- **File-Backed Secrets**: In containerized deployments using the official Docker image, `docker/entrypoint.py` supports loading credentials from mounted secret files via `LAYA_API_KEY_FILE=/run/secrets/api_key`, automatically clearing the path variable from the environment before launching the process.
- **Native JIT Isolation**: The container sets `TORCH_DISABLE_NATIVE_JIT=1` by default to prevent runtime invocation of compilation backends or Triton kernel emission without a local C toolchain.

### 4. Non-Autoregressive Adversarial Considerations
- **Structured Output Boundary**: Unlike generative autoregressive LLMs, Laya is a non-autoregressive encoder-head classifier that produces discrete, bounded predictions (choices, scores, and binary decisions). Because it does not generate freeform token streams, Laya itself cannot be manipulated into emitting arbitrary adversarial text, executing injected code, or leaking internal instructions through text completion.
- **Defense-in-Depth Prompt Screening**: Laya decision hooks (`on_predict_start`) can be deployed as an upstream screening layer to classify incoming inputs for known malicious patterns before forwarding to downstream generative LLMs. Note that statistical prompt classification is a defense-in-depth measure and does not guarantee detection of every novel or obfuscated prompt injection; it should be combined with robust downstream guardrails and least-privilege tool execution.

---

## Automated Security Audits in CI/CD

Veya runs automated quality and security gates on every commit and pull request:
- **Secret Leak Detection**: [`gitleaks`](.github/workflows/security.yml) scans git commit history and the checked-out working tree to prevent credentials, API tokens, and secrets from entering the repository.
- **Dependency Vulnerability Scanning**: [`pip-audit`](.github/workflows/security.yml) audits every
  declared dependency, the core and every extra, as the installs that can exist. Any advisory fails
  the check unless it has been reviewed and listed in
  [`.github/accepted-advisories.toml`](.github/accepted-advisories.toml) as unfixed and unreachable
  through Veya; a listed one fails again once a fix is released. What the package and its image
  ship (the core and `[serve]`) is audited on its own, and nothing in it is ever accepted.
- **Static Analysis**: CodeQL runs automated semantic and security static analysis on every push.
