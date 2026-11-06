"""Measure a local checkpoint pair and exercise the repaired ONNX checkpoint over MCP."""
import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import median
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def frozen_weights(base, candidate, *, allow_encoder_update=False, routing_only=False):
    import torch
    from safetensors import safe_open
    changed, frozen = [], 0
    allowed = ("head.", "type_emb.", "scorer.")
    if allow_encoder_update:
        allowed += ("encoder.",)
    with (safe_open(str(base / "model.safetensors"), framework="pt") as old,
          safe_open(str(candidate / "model.safetensors"), framework="pt") as new):
        added = sorted(set(new.keys()) - set(old.keys()))
        if routing_only:
            assert set(old.keys()) <= set(new.keys())
            assert added and all(name.startswith("choice_adapters.") for name in added), added
        else:
            assert old.keys() == new.keys()
        for name in old.keys():
            before, after = old.get_tensor(name), new.get_tensor(name)
            assert before.dtype == after.dtype and before.shape == after.shape, name
            if torch.equal(before, after):
                frozen += 1
            else:
                assert not routing_only and name.startswith(allowed), name
                changed.append(name)
    assert changed or added
    result = {"identical_tensors": frozen, "updated_tensors": changed}
    if routing_only:
        old_cfg, new_cfg = (json.loads((path / "rl_agent_config.json").read_text(encoding="utf-8"))
                            for path in (base, candidate))
        assert not old_cfg.pop("choice_adapter_width", 0) and new_cfg.pop("choice_adapter_width") > 0
        assert old_cfg.pop("temperature")[1:] == new_cfg.pop("temperature")[1:]
        for cfg in (old_cfg, new_cfg):
            cfg.pop("training", None)
        assert old_cfg == new_cfg
        result["added_tensors"] = added
    return result


def compare_latency(args, cases):
    from laya.agent import Agent
    from laya.onnx_agent import _NativeONNXAgent
    from laya.qualification import _answers, _compare, _percentiles

    sample = []
    for kind in ("choice", "noul", "score"):
        sample.extend([case for case in cases if case["expected"] and
                       next(iter(case["questions"].values()))["type"] == kind][:3])
    if len(sample) != 9:
        raise ValueError("latency comparison needs three labelled cases of each question type")
    result = {}
    common = {"performance_mode": "default", "decision_policy": "canonical"}
    for backend in ("eager", "onnx"):
        with ExitStack() as stack:
            forward_counts = [0, 0]
            agents = []
            for index, (directory, graph) in enumerate(((args.base, args.base_onnx), (args.candidate, args.onnx))):
                if backend == "eager":
                    agent = stack.enter_context(Agent(str(directory), device="cpu", backend="eager", **common))
                    def count(_module, _inputs, index=index):
                        forward_counts[index] += 1
                    stack.enter_context(agent.model.register_forward_pre_hook(count))
                else:
                    agent = _NativeONNXAgent(str(directory), onnx_path=str(graph), **common)
                    stack.callback(setattr, agent, "session", None)
                    session = agent.session
                    class CountSession:
                        def __init__(self, wrapped, index):
                            self.wrapped, self.index = wrapped, index

                        def run(self, *args, **kwargs):
                            forward_counts[self.index] += 1
                            return self.wrapped.run(*args, **kwargs)
                    agent.session = CountSession(session, index)
                agents.append(agent)
            for agent in agents:
                for case in sample:
                    agent.predict(case["state"], case["questions"])
            forward_counts[:] = [0, 0]
            pairs = []
            for repeat in range(3):
                for position, case in enumerate(sample):
                    timings = [0.0, 0.0]
                    start_index = (repeat + position) % 2
                    for index in (start_index, 1 - start_index):
                        start = time.perf_counter()
                        agents[index].predict(case["state"], case["questions"])
                        timings[index] = (time.perf_counter() - start) * 1000
                    pairs.append({"id": case["id"], "repeat": repeat, "first": start_index,
                                  "base_ms": timings[0], "candidate_ms": timings[1]})
            assert forward_counts == [len(pairs)] * 2, forward_counts
            result[backend] = {"pairs": pairs, "forward_counts": forward_counts.copy(),
                               "base_ms": _percentiles([p["base_ms"] for p in pairs]),
                               "candidate_ms": _percentiles([p["candidate_ms"] for p in pairs]),
                               "paired_median_ratio": median(p["candidate_ms"] / p["base_ms"] for p in pairs)}
            if args.routing_only:
                questions = {question["type"]: question for case in sample
                             for question in case["questions"].values()}
                mixed = []
                for index, agent in enumerate(agents):
                    before = forward_counts[index]
                    answers = _answers(agent.predict(sample[0]["state"], questions), questions)
                    assert forward_counts[index] - before == 1
                    mixed.append({"id": "mixed", "answers": {kind: answers[kind] for kind in ("score", "noul")}})
                drift = _compare([mixed[0]], [mixed[1]])
                assert drift["label_flips"] == drift["operational_flag_flips"] == 0, drift
                assert max(drift[key] for key in drift if key.endswith("_drift")) <= .001, drift
                result[backend]["mixed_request"] = {"forward_counts": [1, 1], "protected_answers": drift}
    return result


async def stdio_contract(mode, env, payloads, expected, fresh, fresh_answers):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "laya.mcp.server", "--mode", mode],
                                  env=env, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=90) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert ({"veya_execute"} == names if mode == "code" else
                    {"laya_status", "laya_predict", "laya_predict_batch"} <= names), names

            async def call(name, arguments, code):
                if mode == "code":
                    name, arguments = "veya_execute", {"code": code, "input": arguments}
                wire = (await session.call_tool(name, arguments)).model_dump(mode="json", by_alias=True)
                assert not wire.get("isError", False), wire
                return json.loads("".join(part["text"] for part in wire["content"] if part["type"] == "text"))

            status_code = "async (_, veya) => await veya.status({})"
            status = await call("laya_status", {}, status_code)
            assert status["backend"] == "onnx" and status["loaded"] == [], status
            batch = await call("laya_predict_batch", {"requests": payloads},
                               "async (input, veya) => await veya.predict_batch(input)")
            assert [entry["answers"] for entry in batch["requests"]] == expected
            status = await call("laya_status", {}, status_code)
            assert status["loaded"] == [] and status["model_free_replay"]["enabled"], status
            assert status["decision_cache"]["required"], status
            prediction = "async (input, veya) => await veya.predict(input)"
            before = status["decision_cache"]
            result = await call("laya_predict", fresh, prediction)
            assert result["answers"] == fresh_answers, result
            resident = await call("laya_status", {}, status_code)
            assert resident["loaded"] == [fresh["model"]], resident
            assert resident["checkpoint_policies"] == {fresh["model"]: "canonical"}, resident
            written = resident["decision_cache"]
            assert written["size"] == before["size"] + 1 and written["misses"] == before["misses"] + 1
            assert (await call("laya_predict", fresh, prediction))["answers"] == fresh_answers
            replayed = (await call("laya_status", {}, status_code))["decision_cache"]
            assert replayed["size"] == written["size"] and replayed["misses"] == written["misses"]
            assert replayed["hits"] == written["hits"] + 1
            return {"catalog_size": len(names), "retained_predictions": len(expected),
                    "cold_models_loaded": status["loaded"], "fresh_models_loaded": resident["loaded"],
                    "decision_policy": resident["checkpoint_policies"],
                    "fresh_misses": written["misses"] - before["misses"],
                    "repeat_hits": replayed["hits"] - written["hits"]}


def mcp_smoke(args, cases, artifacts):
    from laya.serve import _close_decision_cache, build_router

    name = "english-repair"
    if args.onnx.parent != args.candidate:
        raise ValueError("keep the export and external tensor files beside the candidate")
    pins = {key.removeprefix("onnx-external/"): value for key, value in artifacts.items()
            if key.startswith(("tokenizer/", "onnx-external/")) or key == "rl_agent_config.json"}
    pins["onnx"] = artifacts["onnx/" + args.onnx.name]
    env = {key: value for key, value in os.environ.items() if not key.startswith("LAYA_")}
    env.update(LAYA_BACKEND="onnx", LAYA_DEVICE="cpu", LAYA_THREADS="4", LAYA_PRELOAD="0",
               LAYA_DECISION_POLICY="canonical", LAYA_EXTRA_MODELS=json.dumps({name: str(args.candidate)}),
               LAYA_ONNX_PATHS=json.dumps({name: str(args.onnx)}), LAYA_SHA256_DIGESTS=json.dumps({name: pins}),
               LAYA_CACHE_REQUIRED="1", LAYA_CACHE_TTL="none", LAYA_CACHE_MAXSIZE="none",
               LAYA_CACHE_FINGERPRINT="repair-" + artifacts["model.safetensors"])
    payloads = [{"model": name, "state": {"message": case["state"]}, "questions": case["questions"]}
                for case in cases[:4]]
    if len({json.dumps(payload, ensure_ascii=False) for payload in payloads}) != 4:
        raise ValueError("MCP verification needs four distinct input payloads")
    with tempfile.TemporaryDirectory(prefix="repair-mcp-") as tmp:
        env["LAYA_CACHE_PATH"] = str(Path(tmp) / "decisions.sqlite")
        with patch.dict(os.environ, env, clear=True):
            with build_router(backend="onnx") as router:
                try:
                    expected = [router.predict(**payload)["answers"] for payload in payloads[:2]]
                    agent = router.load(name)
                    fresh = [agent.predict(payload["state"], payload["questions"])["answers"] for payload in payloads[2:]]
                    manifest = Path(tmp) / "approved.json"
                    router.export_replay_manifest(str(manifest))
                    store_id = router._laya_decision_cache.store_id
                finally:
                    _close_decision_cache(router)
        env.update(LAYA_REPLAY_MANIFEST=str(manifest), LAYA_CACHE_EXPECTED_STORE_ID=store_id)
        return {mode: asyncio.run(stdio_contract(mode, env, payloads[:2], expected, payload, answers))
                for mode, payload, answers in zip(("classic", "code"), payloads[2:], fresh)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "base-onnx", "candidate", "onnx", "corpus", "out"):
        parser.add_argument("--" + name, type=lambda path: Path(path).resolve(), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--allow-encoder-update", action="store_true",
                       help="permit changed encoder weights with identical tensor names, shapes and dtypes")
    modes.add_argument("--routing-only", action="store_true",
                       help="verify added choice adapters and unchanged scores and binary decisions in a mixed request")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new report path")
    for key in tuple(os.environ):
        if key.startswith("LAYA_"):
            del os.environ[key]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32", LAYA_THREADS="4")
    import torch
    from laya.qualification import _artifact_identity, _file_sha, _source_identity, load_document, validate_corpus
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    def identities():
        return {"corpus_sha256": _file_sha(args.corpus), "script_sha256": _file_sha(Path(__file__)),
                "source_sha256": _source_identity(), "base_artifacts": _artifact_identity(args.base, args.base_onnx),
                "candidate_artifacts": _artifact_identity(args.candidate, args.onnx)}
    before = identities()
    if args.routing_only:
        base_metadata, candidate_metadata = ({name: digest for name, digest in before[key].items()
                                               if name.startswith(("encoder/", "tokenizer/"))}
                                              for key in ("base_artifacts", "candidate_artifacts"))
        if base_metadata != candidate_metadata:
            raise ValueError("routing-only verification requires identical encoder and tokenizer metadata")
    cases = [case for case in validate_corpus(load_document(args.corpus))["cases"] if case["split"] == "evaluation"]
    report = {"schema": "veya-checkpoint-verification/1", **before,
              "weight_comparison": frozen_weights(args.base, args.candidate,
                                                  allow_encoder_update=args.allow_encoder_update,
                                                  routing_only=args.routing_only),
              "cpu_threads": 4, "latency": compare_latency(args, cases),
              "mcp": mcp_smoke(args, cases, before["candidate_artifacts"])}
    if identities() != before:
        raise RuntimeError("checkpoint, corpus or source identity changed during verification")
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Saved", args.out, flush=True)


if __name__ == "__main__":
    main()
