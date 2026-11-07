"""Compare ordinary cold durable hits with verified model-free replay on a local checkpoint.

No downloads: --checkpoint must already contain the pretrained English checkpoint.
Run after source changes and other checkpoint benchmarks have finished.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STATE = "A customer was billed twice. Please review the duplicate payment today."
QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does this request need immediate attention?"}}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources():
    paths = sorted((ROOT / "laya").rglob("*.py")) + [Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def peak_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                          "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                          "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        read.restype = wintypes.BOOL
        if read(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


def distribution(values):
    ordered = sorted(values)
    return {"count": len(values), **{name: ordered[min(len(ordered) - 1, math.ceil(len(ordered) * cut) - 1)]
                                    for name, cut in (("p50", .50), ("p95", .95), ("p99", .99))}}


def child(mode, checkpoint, database, manifest, store_id):
    denied = []
    if mode == "verified":
        import importlib.abc
        class NoInference(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.split(".")[0] in ("torch", "transformers", "onnxruntime", "tokenizers", "onnx"):
                    denied.append(fullname)
                    raise AssertionError("verified replay imported " + fullname)
        sys.meta_path.insert(0, NoInference())
    from veya import DecisionCache, Router
    forwards = 0
    if mode == "ordinary":
        from veya.agent import Agent
        original = Agent._forward
        def counted(*args, **kwargs):
            nonlocal forwards
            forwards += 1
            return original(*args, **kwargs)
        Agent._forward = counted
    started = time.perf_counter_ns()
    with DecisionCache(database, required=True, fingerprint="replay-benchmark-v1", maxsize=None,
                       expected_store_id=store_id) as cache:
        router = Router(models={"english": checkpoint}, hooks=[cache], device="cpu", performance_mode="default",
                        **({"replay_manifest": manifest} if mode == "verified" else {}))
        result = router.predict(STATE, QUESTIONS, model="english")
        operation_ms = (time.perf_counter_ns() - started) / 1e6
        loaded = router.loaded
        info = cache.cache_info()
        router.unload()
    assert not denied, denied
    print(json.dumps({"mode": mode, "operation_ms": operation_ms, "peak_memory_bytes": peak_memory(),
                      "answers": result["answers"], "loaded_checkpoints": loaded, "forwards": forwards,
                      "blocked_import_attempts": denied,
                      "inference_libraries": sorted(name for name in ("torch", "transformers", "onnxruntime", "tokenizers", "onnx")
                                                    if name in sys.modules), "cache": info}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warm-hits", type=int, default=500)
    parser.add_argument("--child", choices=("ordinary", "verified"))
    parser.add_argument("--database")
    parser.add_argument("--manifest")
    parser.add_argument("--store-id")
    args = parser.parse_args()
    if args.child:
        child(args.child, str(args.checkpoint), args.database, args.manifest, args.store_id)
        return 0
    if args.checkpoint is None or args.out is None or args.repeats < 1 or args.warm_hits < 1:
        parser.error("checkpoint/out and positive repeats/warm-hits are required")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        parser.error("checkpoint must be an existing local weight directory")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    source_hashes = sources()
    from veya import DecisionCache, Router
    import torch
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="veya-verified-replay-", dir=ROOT / ".venv") as directory:
        database = str(Path(directory) / "decisions.sqlite")
        manifest = str(Path(directory) / "policy.json")
        with DecisionCache(database, required=True, fingerprint="replay-benchmark-v1", maxsize=None) as cache:
            with Router(models={"english": str(checkpoint)}, hooks=[cache], device="cpu", performance_mode="default") as router:
                expected = router.predict(STATE, QUESTIONS, model="english")["answers"]
                agent = router.load("english")
                runtime = {"device": str(agent.device), "backend": agent.backend,
                           "dtype": str(agent.dtype), "amp_enabled": agent.amp_enabled,
                           "model_parameter_dtype": str(next(agent.model.parameters()).dtype)}
                del agent
                router.export_replay_manifest(manifest)
                identity = cache.store_id
        runs = []
        for repeat in range(args.repeats):
            for mode in (("ordinary", "verified") if repeat % 2 == 0 else ("verified", "ordinary")):
                command = [sys.executable, str(Path(__file__).resolve()), "--child", mode,
                           "--checkpoint", str(checkpoint), "--database", database,
                           "--manifest", manifest, "--store-id", identity]
                start = time.perf_counter_ns()
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180, check=True)
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                row = json.loads(result.stdout.splitlines()[-1])
                assert row["answers"] == expected and row["forwards"] == 0
                assert row["cache"]["hits"] == 1 and row["cache"]["errors"] == 0
                assert bool(row["loaded_checkpoints"]) == (mode == "ordinary")
                row.update(repeat=repeat, process_wall_ms=wall_ms)
                runs.append(row)
        warm = {}
        for mode in ("ordinary", "verified"):
            with DecisionCache(database, required=True, fingerprint="replay-benchmark-v1", maxsize=None,
                               expected_store_id=identity) as cache:
                with Router(models={"english": str(checkpoint)}, hooks=[cache], device="cpu", performance_mode="default",
                            **({"replay_manifest": manifest} if mode == "verified" else {})) as router:
                    router.predict(STATE, QUESTIONS, model="english")
                    samples = []
                    for _ in range(args.warm_hits):
                        start = time.perf_counter_ns()
                        result = router.predict(STATE, QUESTIONS, model="english")
                        samples.append((time.perf_counter_ns() - start) / 1000)
                        assert result["answers"] == expected
                    warm[mode] = {"latency_us": distribution(samples), "samples_us": samples,
                                  "cache": cache.cache_info()}
        assert source_hashes == sources(), "runtime source changed during benchmark; rerun after source freeze"
        artifact_paths = [checkpoint / "model.safetensors", checkpoint / "rl_agent_config.json"]
        if (checkpoint / "encoder" / "config.json").is_file():
            artifact_paths.append(checkpoint / "encoder" / "config.json")
        artifact_paths += sorted((checkpoint / "tokenizer").glob("*"))
        artifacts = {path.relative_to(checkpoint).as_posix(): sha256(path) for path in artifact_paths if path.is_file()}
        summary = {mode: {metric: distribution([row[metric] for row in runs if row["mode"] == mode])
                          for metric in ("process_wall_ms", "operation_ms", "peak_memory_bytes")}
                   for mode in ("ordinary", "verified")}
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                  "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                  "torch": torch.__version__, "threads": 4, "runtime": runtime,
                  "checkpoint": str(checkpoint), "artifacts": artifacts,
                  "source_sha256": source_hashes, "repeats": args.repeats, "runs": runs,
                  "summary": summary, "warm": warm, "outputs_equal": True,
                  "method": "Alternating fresh Python processes; same durable decision and existing CPU fp32 checkpoint. "
                            "Parent process wall latency includes interpreter/library imports and cleanup. Child operation "
                            "latency begins after required imports and includes opening the store, Router construction and "
                            "first answer. Ordinary replay loads the model but performs no forward; verified replay actively "
                            "rejects inference-library imports. Peak memory is the process high-water working set, including "
                            "cleanup. Warm hits run separately in the parent; its unrelated resident libraries remain loaded. "
                            "Nearest-rank percentiles; tiny cold tail sample, filesystem caches and workstation load uncontrolled. "
                            "No uncached-inference speed or general accuracy claim."}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
