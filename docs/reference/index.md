# Python API

These pages are generated from the docstrings in `veya/`, so they change with the code. Apart
from `ONNXAgent` and `veya.train`, every name below can be imported from the top-level package,
for example `from veya import Router`.

- [Agent](agent.md): `Agent` and `load` run one checkpoint; `ONNXAgent` runs an exported ONNX
  model.
- [Router](router.md): `Router` picks the checkpoint for each request; `RouteDecision` records
  the choice.
- [Helpers](helpers.md): language detection, email cleaning, question presets, decision
  consistency (`decision_margins`, `DecisionCache`), shortlisting and calibration utilities.
- [Fine-tuning](train.md): `finetune` and `TrainConfig` from `veya.train`, imported from that
  module rather than the top-level package, and the pieces they are built from.
- [LangChain components](langchain.md): `VeyaRouter`, `VeyaGuardrail`, `VeyaTriage` and
  `VeyaEvaluator`.

Prediction hooks have a hand-written [API reference](../hooks/api.md) with the rest of the
[hooks guide](../hooks/index.md).
