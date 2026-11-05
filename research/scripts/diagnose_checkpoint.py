"""Diagnose an existing local checkpoint without changing its weights, corpus or gates.

Reports the canonical policy, fixed cyclic option rotations, the fixed content-free
prior correction from https://proceedings.mlr.press/v139/zhao21c.html, and the actual
opt-in context-balanced-v1 policy. Labels are used only after inference for scoring.
This is development evidence, not release qualification or an unseen accuracy estimate.
"""
import argparse
from datetime import datetime, timezone
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _answer(question, probabilities):
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": float(probabilities[1])}
    labels = list(question["criteria"]) if kind == "choice" else list(map(str, range(len(probabilities))))
    answer = {"type": kind, "probabilities": dict(zip(labels, map(float, probabilities)))}
    if kind == "choice":
        answer["choice"] = labels[int(probabilities.argmax())]
    else:
        answer["score"] = float(sum(index * value for index, value in enumerate(probabilities)))
    return answer


def _summary(cases, results):
    from laya import qualification
    rows, errors, boundaries = [], [], []
    for case, result in zip(cases, results):
        answers = qualification._answers(result, case["questions"])
        rows.append({"id": case["id"], "result": result})
        for qid, answer in answers.items():
            if answer["margin"] <= 0.05:
                boundaries.append({"id": case["id"], "question": qid, "margin": answer["margin"]})
            if qid in case["expected"]:
                expected = case["expected"][qid]
                actual = answer["score"] if answer["score"] is not None else answer["label"]
                correct = abs(actual - expected) <= 0.25 if answer["score"] is not None else actual == expected
                if not correct:
                    errors.append({"id": case["id"], "question": qid, "actual": actual, "expected": expected})
    return {"metrics": qualification._quality(cases, results), "observations": rows, "errors": errors,
            "boundary_observations_at_0.05": boundaries,
            "boundary_cases_at_0.05": len({row["id"] for row in boundaries})}


def _observe_canonical(agent, cases, questions):
    raw_logits = []
    def capture(_module, _args, outputs):
        raw_logits.extend(outputs[0].detach().cpu().tolist())
    started = time.perf_counter()
    with agent.model.register_forward_hook(capture):
        results = agent.predict_batch([case["state"] for case in cases], questions,
                                      batch_size=4, sort_by_length=False)
    elapsed = time.perf_counter() - started
    if len(raw_logits) != len(cases) * len(questions):
        raise RuntimeError("canonical forward rows do not match the requested state/question order")
    internal = {qid: agent._to_internal(q) for qid, q in questions.items()}
    stats = []
    for case in cases:
        items = agent._encode_state(case["state"], list(questions), internal)
        for qid, item in zip(questions, items):
            stats.append({"id": case["id"], "question": qid,
                "raw_logits": raw_logits[len(stats)][:len(item["markers"])],
                "sequence_tokens": len(item["ids"]), "marker_positions": item["markers"],
                "option_stats": item["options"], "state_stats": item["state_stats"],
                "decoded_sequence": agent.tok.decode(item["ids"])})
    return results, elapsed, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "research/evals/qualification-v1.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    checkpoint, corpus_path, target = args.checkpoint.resolve(), args.corpus.resolve(), args.out.resolve()
    if args.threads < 1:
        parser.error("threads must be positive")
    if target.exists() or target.is_relative_to(checkpoint):
        parser.error("choose a new output file outside the checkpoint; existing evidence is never overwritten")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
    import numpy as np
    import torch
    from laya import qualification
    from laya.agent import Agent
    from laya.decision_policy import CONTENT_FREE_PROBES, MAX_BALANCED_OPTIONS, _correct
    from laya.inference import _option_count
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    def identities():
        script = Path(__file__).resolve()
        sources = {"laya/" + name: digest for name, digest in qualification._source_identity().items()}
        sources[script.relative_to(ROOT).as_posix()] = qualification._file_sha(script)
        return {"source_sha256": sources, "corpus_file_sha256": qualification._file_sha(corpus_path),
                "artifacts": qualification._artifact_identity(checkpoint)}
    before = identities()
    required = {"encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if not required <= before["artifacts"].keys():
        parser.error("the diagnostic requires a complete local encoder and tokenizer")
    tokenizer = qualification.load_document(checkpoint / "tokenizer/tokenizer_config.json")
    if (tokenizer.get("tokenizer_class") in (None, "TokenizersBackend")
            or isinstance(tokenizer.get("extra_special_tokens"), list)):
        parser.error("tokenizer metadata needs normalization; prepare a checkpoint copy before diagnosis")
    corpus = qualification.validate_corpus(qualification.load_document(corpus_path))
    cases = [case for case in corpus["cases"] if case["split"] == "evaluation"]
    groups = {}
    for index, case in enumerate(cases):
        if any(_option_count(q) > MAX_BALANCED_OPTIONS for q in case["questions"].values()):
            parser.error("diagnostic policies support at most %d options" % MAX_BALANCED_OPTIONS)
        key = json.dumps(case["questions"], ensure_ascii=False, separators=(",", ":"))
        groups.setdefault(key, []).append(index)
    names = ("canonical", "cyclic", "canonical_prior_corrected", "cyclic_prior_corrected", "context-balanced-v1")
    results = {name: [{"answers": {}} for _ in cases] for name in names}
    evidence = []
    with Agent(str(checkpoint), device="cpu", backend="eager", performance_mode="default",
               decision_policy="canonical") as agent:
        for indices in groups.values():
            questions = cases[indices[0]]["questions"]
            group = [cases[index] for index in indices]
            states = [case["state"] for case in group]
            canonical, elapsed, token_stats = _observe_canonical(agent, group, questions)
            for index, result in zip(indices, canonical):
                results["canonical"][index] = result
            detail = {"case_ids": [case["id"] for case in group], "canonical_elapsed_s": elapsed,
                      "canonical_token_stats": token_stats, "questions": {}}
            for qid, question in questions.items():
                count = _option_count(question)
                initial = list(question.get("option_order", range(count)))
                rotations = [initial[shift:] + initial[:shift] for shift in range(count)]
                # Record every order for three-way choice diagnostics only. The
                # tested policies still use the fixed canonical/cyclic rows.
                extra = ([list(order) for order in permutations(initial) if list(order) not in rotations]
                         if question["type"] == "choice" and count == 3 else [])
                variants = {index: dict(question, option_order=order)
                            for index, order in enumerate(rotations + extra)}
                started = time.perf_counter()
                with torch.no_grad():
                    rows = agent._infer_batch(states + list(CONTENT_FREE_PROBES), variants,
                        batch_size=max(1, 64 // len(variants)), sort_by_length=False, full_precision=True)
                all_probabilities = np.asarray([[row["answers"][index]["_decision_probabilities"]
                    for index in variants] for row in rows], dtype=np.float64)
                probabilities = all_probabilities[:, :count]
                prior_rows = probabilities[len(states):]
                prior = prior_rows.mean(axis=(0, 1))
                canonical_prior = prior_rows[:, 0].mean(axis=0)
                detail["questions"][qid] = {"rotations": rotations,
                    "elapsed_s": time.perf_counter() - started, "probe_probabilities": prior_rows.tolist(),
                    "canonical_prior": canonical_prior.tolist(), "cyclic_prior": prior.tolist(),
                    "observed_probabilities": probabilities[:len(states)].tolist(),
                    "usage": [row["usage"] for row in rows]}
                if extra:
                    detail["questions"][qid].update(noncyclic_orders=extra,
                        noncyclic_observations=all_probabilities[:len(states), count:].tolist(),
                        noncyclic_probes=all_probabilities[len(states):, count:].tolist())
                for index, observed in zip(indices, probabilities):
                    mean = observed.mean(axis=0)
                    for name, values in (("cyclic", mean),
                        ("canonical_prior_corrected", _correct(observed[0], canonical_prior)),
                        ("cyclic_prior_corrected", _correct(mean, prior))):
                        results[name][index]["answers"][qid] = _answer(question, values)
            evidence.append(detail)
    with Agent(str(checkpoint), device="cpu", backend="eager", performance_mode="default",
               decision_policy="context-balanced-v1") as agent:
        for detail, indices in zip(evidence, groups.values()):
            started = time.perf_counter()
            observed = agent.predict_batch([cases[index]["state"] for index in indices],
                                           cases[indices[0]]["questions"], batch_size=4, sort_by_length=False)
            detail["production_policy_elapsed_s"] = time.perf_counter() - started
            for index, result in zip(indices, observed):
                results["context-balanced-v1"][index] = result
    after = identities()
    if before != after:
        raise RuntimeError("source, corpus or checkpoint identity changed during diagnosis")
    report = {"schema": "veya-checkpoint-diagnosis/1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "corpus_id": corpus["id"], "corpus_path": str(corpus_path),
        **before, "identity_changed_during_run": False, "runtime": {"python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "device": "cpu", "precision": "float32"},
        "probes": list(CONTENT_FREE_PROBES), "evidence": evidence,
        "policies": {name: _summary(cases, rows) for name, rows in results.items()},
        "production_qualified": False,
        "limits": "Development diagnosis, not qualification. Labels are scoring-only. Fixed cyclic rotations and "
                  "fixed three-probe prior correction have no fitted hyperparameters. Corrected confidence is not "
                  "a calibrated probability. A semantic policy change requires a new cache policy identity and "
                  "independent validation. Boundary coverage is measured on returned probabilities at margin 0.05; "
                  "no gate, original corpus, model weight or earlier evidence is rewritten."}
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    for name, summary in report["policies"].items():
        print(name, summary["metrics"], "errors", len(summary["errors"]),
              "boundary cases", summary["boundary_cases_at_0.05"], flush=True)
    print("Saved", target, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
