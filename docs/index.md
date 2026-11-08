---
title: Veya documentation
description: Run a local AI decision engine with Python, TypeScript, Java and .NET SDKs, HTTP and MCP tools, and configurable decision cache retention.
---

# Veya: local decisions with configurable retention

Veya classifies, scores and answers yes/no questions over text or JSON in a single
forward pass. It runs on your hardware and can replay stored decisions for exact repeat
requests. This independent fork of Laya adds cache retention policies, decision margins
and optimizations around inference.

Start with [installation and a first decision](https://github.com/tgundhus/Veya#install),
then choose an [SDK or local MCP interface](integration.md).

| Need | Guide |
|---|---|
| Install and run the Python engine | [Using Veya](guide.md) |
| Connect another application | [SDKs and integration](integration.md) |
| Configure persistent decisions and retention | [Decision consistency](consistency.md) |
| Deploy HTTP or local MCP | [Deployment](production.md), [HTTP API](http-api.md), [CLI and MCP](cli-mcp.md) |
| Work with schemas and confidence policies | [Structured decisions](structured.md), [Questions and answers](questions-and-answers.md) |
| Check accuracy and performance | [Benchmarks](benchmarks.md), [Limits](limits.md), [October review](reports/production-review.md) |
| Use containers | [Docker](docker.md), [ARM64 and DGX Spark](docker-platforms.md) |
| Look up Python methods | [API reference](reference/index.md) |

The [source repository](https://github.com/tgundhus/Veya) includes SDKs, examples,
regression suites and raw benchmark results. The pretrained checkpoints remain upstream
Veya models. Cache replay fixes repeatability while an entry remains valid; it does not
establish that a decision is correct.
