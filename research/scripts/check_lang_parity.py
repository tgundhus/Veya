"""Language detection of two checkouts on real text: the same result for every input, and the time it takes.

The corpus is every utterance of the mteb/amazon_massive_intent test split in all 51 languages,
long texts made by joining utterances of one language (around 2,000 characters), and each
utterance as the dict state `{"utterance": ...}` the Router also sees. Each checkout runs
`veya.lang.analyse` over it in its own subprocess; the results are compared one by one.

    git worktree add ../laya-main main
    python research/scripts/check_lang_parity.py --base ../laya-main --out lang_parity.json
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATASET = "mteb/amazon_massive_intent"

WORKER = r"""
import json, sys, time
sys.path.insert(0, %(root)r)
from veya.lang import analyse
import veya
corpus = json.load(open(%(corpus)r, encoding="utf-8"))
out, t_cpu = [], {}
for kind, items in corpus.items():
    t = time.thread_time()
    for x in items:
        out.append(json.dumps(analyse(x), sort_keys=True, ensure_ascii=False))
    t_cpu[kind] = time.thread_time() - t
json.dump({"laya_file": veya.__file__, "results": out, "cpu_s": t_cpu}, open(%(result)r, "w", encoding="utf-8"))
"""


def corpus(per_long=40, seed=5):
    from datasets import get_dataset_config_names, load_dataset

    rng = random.Random(seed)
    short, long_, dicts = [], [], []
    for lang in sorted(n for n in get_dataset_config_names(DATASET) if n != "default"):
        texts = [r["text"] for r in load_dataset(DATASET, lang, split="test")]
        short.extend(texts)
        dicts.extend({"utterance": t} for t in texts[:500])
        for _ in range(per_long):
            picked, size = [], 0
            while size < 2000:
                t = rng.choice(texts)
                picked.append(t)
                size += len(t) + 2
            long_.append(". ".join(picked))
    return {"utterances": short, "long texts": long_, "dict states": dicts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default=ROOT)
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds per checkout, alternating")
    parser.add_argument("--out")
    args = parser.parse_args()
    tmp = tempfile.mkdtemp(prefix="laya_lang_")
    data = corpus()
    path = os.path.join(tmp, "corpus.json")
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    n = {k: len(v) for k, v in data.items()}
    print("corpus:", n, flush=True)
    runs = {"base": [], "head": []}
    for rnd in range(args.rounds):
        for which in (("base", "head") if rnd % 2 == 0 else ("head", "base")):
            root = os.path.abspath(getattr(args, which))
            result = os.path.join(tmp, "%s_%d.json" % (which, rnd))
            subprocess.run([sys.executable, "-I", "-c", WORKER % {"root": root, "corpus": path, "result": result}],
                           check=True)
            got = json.load(open(result, encoding="utf-8"))
            assert got["laya_file"].startswith(root), got["laya_file"]
            runs[which].append(got)
    a, b = runs["base"][0]["results"], runs["head"][0]["results"]
    kinds = [k for k, v in data.items() for _ in v]
    items = [x for v in data.values() for x in v]
    differ = [{"kind": k, "input": str(x)[:120], "base": ra, "head": rb}
              for k, x, ra, rb in zip(kinds, items, a, b) if ra != rb]
    cpu = {w: {k: round(min(r["cpu_s"][k] for r in rs), 3) for k in data} for w, rs in runs.items()}
    report = {"meta": {"platform": platform.platform(), "machine": platform.machine(),
                       "python": platform.python_version(), "dataset": DATASET, "inputs": n, "rounds": args.rounds},
              "compared": len(a), "identical": len(a) - len(differ), "differ": differ[:50],
              "digest_base": hashlib.sha256("\n".join(a).encode()).hexdigest()[:16],
              "digest_head": hashlib.sha256("\n".join(b).encode()).hexdigest()[:16],
              "cpu_s_best_of_rounds": cpu,
              "speedup": {k: round(cpu["base"][k] / cpu["head"][k], 3) for k in data},
              "us_per_input": {w: {k: round(cpu[w][k] / n[k] * 1e6, 2) for k in data} for w in cpu}}
    print(json.dumps({k: v for k, v in report.items() if k != "differ"}, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
