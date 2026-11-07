"""Check that fp32, int8 and ONNX runs of one checkpoint share one DecisionCache.

The cache's fingerprint is the checkpoint id, revision, calibration temperatures, token budget
and Laya version; the backend and precision are not in it. So a decision stored by the PyTorch
fp32 Agent is replayed by an int8 Agent and by an ONNXAgent of the same checkpoint, loaded by the
same id or path, even though each backend on its own answers slightly differently.

The check loads the three backends, prints whether their own answers match, then runs them in
turn over one SQLite file and asserts that the second and third replay the first one's decision
without running their model.

    python research/scripts/check_backend_sharing.py --model convaiinnovations/laya \\
        --onnx english.onnx          # from scripts/export_onnx.py; its .int8.onnx works too
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "tone": {"type": "choice", "instructions": "What is the customer's tone?",
             "criteria": {"calm": "calm and polite", "angry": "angry or upset"}},
}
STATE = "Hi, we were billed twice for March. Please refund the duplicate today."


def int8_clone(agent):
    """The same Agent with its encoder's nn.Linear layers dynamically quantized to int8."""
    import copy

    import torch

    names = {n for n, m in agent.model.named_modules() if n.startswith("encoder.") and isinstance(m, torch.nn.Linear)}
    clone = copy.copy(agent)
    clone.model = torch.ao.quantization.quantize_dynamic(agent.model, names, dtype=torch.qint8)
    return clone


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="checkpoint id or path, the same for every backend")
    parser.add_argument("--onnx", required=True, help="ONNX export of that checkpoint")
    args = parser.parse_args()

    from veya import DecisionCache
    from veya.agent import Agent
    from veya.consistency import _fingerprint
    from veya.onnx_agent import ONNXAgent

    fp32 = Agent(args.model, device="cpu")
    backends = [("torch-fp32", fp32), ("torch-int8", int8_clone(fp32)),
                ("onnx", ONNXAgent(args.model, onnx_path=args.onnx))]
    prints = [_fingerprint(agent) for _, agent in backends]
    assert all(p == prints[0] for p in prints), prints
    print("one fingerprint for all three backends")
    own = [agent.predict(STATE, QUESTIONS)["answers"] for _, agent in backends]
    print("their own answers are %s" % ("identical" if all(a == own[0] for a in own) else "not identical"))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "decisions.sqlite")
        replayed = []
        for name, agent in backends:
            cache = DecisionCache(path)
            agent.hooks = [cache]
            t0 = time.perf_counter()
            replayed.append(agent.predict(STATE, QUESTIONS)["answers"])
            info = cache.cache_info()
            print("%-10s %9.2f ms  %s" % (name, (time.perf_counter() - t0) * 1000,
                                          "stored" if info["misses"] else "replayed"))
            cache.close()
        assert all(a == replayed[0] for a in replayed), "every backend should replay the first decision"
        print("int8 and ONNX replayed the fp32 decision")


if __name__ == "__main__":
    main()
