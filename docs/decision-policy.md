---
title: Context-balanced decisions
description: Reduce checkpoint preference for answers without evidence, using an optional local inference policy.
---

# Context-balanced decisions

The default `canonical` policy runs the checkpoint's usual decision path. An
experimental `context-balanced-v1` policy reduces sensitivity to option placement and
to preferences the checkpoint expresses even when no relevant state is present.
It uses the same weights and tokenizer. It adds inference work, so measure its
quality and latency on your own cases before using it for new decisions.

The original fixture's four errors were corrected, but independent controls
showed new binary and score errors. Keep the canonical default unless the new
policy earns acceptance on your deployment's cases.

```python
from laya import Router

with Router(models={"english": "/models/english"},
            agent_kwargs={"decision_policy": "context-balanced-v1"}) as router:
    result = router.predict(state, questions, model="english")
```

`Agent`, `ONNXAgent` and `laya.load` accept the same keyword. When it is omitted,
the checkpoint configuration's `decision_policy` is used, with `canonical` as
the default. Use `decision_policy="canonical"` to explicitly request the ordinary
path. HTTP and local MCP accept `LAYA_DECISION_POLICY=context-balanced-v1` in the
server environment. Docker Compose forwards it; the Nix service option is
`decisionPolicy`. Standalone `veya-onnx` supports the policy without Torch.
The native TypeScript inference engine keeps its canonical policy; HTTP clients
can consume the policy answers from a Python or standalone ONNX server.

## What the policy changes

For a question with `k` options, Veya runs all `k` cyclic rotations, maps results
back to the caller's labels, and averages their temperature-scaled probabilities.
Each option occupies each slot once. This reduces a positional effect without
claiming invariance to arbitrary permutations. The same rotations are evaluated
on three fixed content-free inputs: empty text, `N/A` and `[MASK]`. The regular
tokenizer handling still sanitizes any reserved mask token in state text.

Veya divides each averaged real-input probability by its averaged content-free
prior and normalizes the resulting vector. The calculation uses full precision
probabilities, with a float64 floor for underflow. The method is inspired by
[contextual calibration](https://proceedings.mlr.press/v139/zhao21c.html); that
paper evaluated few-shot language models, so its accuracy results do not transfer
to these decision checkpoints. The rotations and probes are fixed across choice,
noul and score. They never depend on expected answers or neighbouring requests.
Removing a prior can also remove a useful preference; this policy needs its own
validation on representative inputs.

The corrected vector supplies the label, yes probability or expected ordinal
score. `decision_policy: "context-balanced-v1"` identifies the result and
`confidence_calibrated: false` marks the corrected confidence as unverified.
The checkpoint's original confidence binning is not applied to corrected scores.
`action.act_probability` is the mean of the original actions on the real-input
rotations; it is not an evidence-corrected action. An existing confidence gate
can still flag an answer, but a threshold fitted to canonical outputs should
not be reused without validating it on corrected outputs.

## Work and cache identity

The policy supports up to 16 options per question. Each real state costs one row
per rotation. The three content-free probes are shared across states with the
same questions in a call. Forward batches are bounded; the additional work
still makes a fresh decision slower. `usage.input_tokens` counts the extra rows,
with shared `policy_probe_tokens` charged once to the first result. Code Mode
checks the expanded work before invoking a resident local model. HTTP applies
its forward token limit to both rotations and probes. A configured limit smaller
than one effective token window refuses the corrected call before inference.

Canonical cache keys are preserved. Corrected decisions use a separate key
namespace even if the operator supplied the same explicit cache fingerprint.
Re-export a [verified replay manifest](replay.md) for the new policy before
using its cold replay path. Switching policies intentionally changes new answers;
retained decisions within each policy continue to replay exactly, with usage from
the recorded computation.

Use a [qualification matrix](release-qualification.md) with `decision_policy`
on each applicable cell. Compare Torch and ONNX under the same policy; a saved
canonical baseline is a different decision identity. The
[checkpoint investigation](reports/checkpoint-qualification-repair-2026-11-05.md)
records observed accuracy changes, costs and the unchanged coverage gate.
