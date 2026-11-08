"""Measure FIFO/LRU through a real Router on the same frozen support requests.

python research/scripts/bench_cache_workflow.py --checkpoint /local/multilingual --out workflow.json
No downloads or hosted inference. Each trial starts with an empty bounded memory cache.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/benchmarks/feishu_zh"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
from audit import audit_run, load_cases  # noqa: E402
from prompts import requests_for  # noqa: E402
import torch  # noqa: E402
from veya import Agent, DecisionCache, Router  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "research/results/matched_feishu_veya_cpu_20261107")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.capacity < 1 or args.requests < 1 or args.repeats < 2:
        parser.error("capacity/requests must be positive and repeats >=2")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cases, protocol = load_cases()
    audit_run(args.reference)
    metadata = json.loads((args.reference / "metadata.json").read_text(encoding="utf-8"))
    weight_hash = hashlib.file_digest((args.checkpoint / "model.safetensors").open("rb"), "sha256").hexdigest()
    assert weight_hash == metadata["checkpoint_model_sha256"], "Use the matched run's exact weights"
    rows = [json.loads(line) for line in (args.reference / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    reference = {row["id"]: row["response"]["answers"] for row in rows
                 if row["mode"] == "choice" and row["repeat"] == 0 and row["status"] == "ok"}
    assert len(reference) == len(cases) == 64
    requests = [requests_for(case)["choice"] for case in cases]
    indices = random.Random(20261107).choices(range(len(cases)),
                                             weights=[1 / (i + 1) ** 1.1 for i in range(len(cases))],
                                             k=args.requests)
    agent = Agent(str(args.checkpoint.resolve()), device="cpu")
    agent.predict(**requests[0])
    trials = {"fifo": [], "lru": []}
    for repeat in range(args.repeats):
        for policy in (("fifo", "lru") if repeat % 2 == 0 else ("lru", "fifo")):
            cache = DecisionCache(maxsize=args.capacity, eviction=policy)
            with Router(hooks=[cache]) as router:
                router.attach("multilingual", agent)
                samples, changes, forwards = [], 0, [0]
                original = agent._forward

                def counted(*values, **kwargs):
                    forwards[0] += 1
                    return original(*values, **kwargs)

                agent._forward = counted
                try:
                    started = time.perf_counter()
                    for index in indices:
                        tick = time.perf_counter()
                        result = router.predict(**requests[index], model="multilingual")
                        samples.append((time.perf_counter() - tick) * 1000)
                        changes += result["answers"] != reference[cases[index]["id"]]
                    seconds = time.perf_counter() - started
                finally:
                    agent._forward = original
                info = cache.cache_info()
                assert forwards[0] == info["misses"] and changes == 0, (forwards, info, changes)
                trials[policy].append({"repeat": repeat, "wall_seconds": seconds,
                                       "forward_passes": forwards[0], "changed_answer_payloads": changes,
                                       "cache": info, "samples_ms": samples})
                print(policy, repeat, round(seconds, 3), "seconds", forwards[0], "forwards", flush=True)
            cache.close()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "torch": torch.__version__, "threads": torch.get_num_threads(),
              "parameters": vars(args) | {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
              "protocol": protocol, "case_indices": indices,
              "method": "Same seeded Zipf trace, real multilingual checkpoint, choice mode, empty cache per trial. "
                        "Alternating policy order, one warmed shared agent, 2 CPU threads. "
                        "Uncontrolled workstation load; no Jev or cross-device latency comparison.",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("veya/consistency.py", "veya/router.py", "veya/agent.py",
                                             "research/scripts/bench_cache_workflow.py")},
              "reference_metadata_sha256": hashlib.sha256((args.reference / "metadata.json").read_bytes()).hexdigest(),
              "reference_raw_sha256": hashlib.sha256((args.reference / "raw.jsonl").read_bytes()).hexdigest(),
              "checkpoint_model_sha256": weight_hash,
              "policies": {policy: {"trials": samples,
                                     "median_wall_seconds": statistics.median(s["wall_seconds"] for s in samples),
                                     "forward_passes_per_trial": samples[0]["forward_passes"]}
                           for policy, samples in trials.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
