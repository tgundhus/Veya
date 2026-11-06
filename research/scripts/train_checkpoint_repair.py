"""Train the existing decision head on local contrast examples with a frozen encoder."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def training_rows(groups):
    for group in groups:
        question = {key: group[key] for key in ("type", "instructions", "criteria") if key in group}
        for state, target in group["rows"]:
            yield {"state": state, "questions": {"decision": question}, "expected": {"decision": target},
                   "family": group["family"]}


def extract(agent, rows, *, rotate_kinds=()):
    import torch
    from torch.nn.functional import pad
    from laya.common import collate_items
    from laya.train import encode_item, items_from_rows

    max_len, head_len = (agent.cfg[key] for key in ("max_len", "head_max_len"))
    items, skipped = items_from_rows(agent.tok, rows, max_len, head_len)
    expected = sum(len(row["expected"]) for row in rows)
    if skipped or len(items) != expected:
        raise ValueError("invalid training inputs: %r, expected %d, encoded %d" % (skipped, expected, len(items)))
    device = next(agent.model.parameters()).device
    groups = defaultdict(list)
    for item in items:
        k = item["k"]
        if k != (2 if item["qtype"] == 2 else 3):
            raise ValueError("this English repair recipe uses three-way choice/score and binary noul")
        for offset in range(k if item["qtype"] in rotate_kinds else 1):
            order = [(i + offset) % k for i in range(k)]
            encoded = encode_item(agent.tok, item, max_len, head_len, order)
            encoded["levels"] = order
            groups[k].append(encoded)
    output = []
    for k, group in groups.items():
        features, masks, markers, levels, targets, kinds = [], [], [], [], [], []
        width = max(len(item["ids"]) for item in group)
        with torch.no_grad():
            for start in range(0, len(group), 12):
                chunk = group[start:start + 12]
                batch = collate_items([chunk], agent.tok.pad_token_id)
                batch = {key: value.to(device) for key, value in batch.items() if key != "meta"}
                hidden = agent.model.encoder(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"]).last_hidden_state
                extra = width - hidden.shape[1]
                features.append(pad(hidden, (0, 0, 0, extra)))
                masks.append(pad(batch["attention_mask"], (0, extra)))
                markers.append(batch["marker_pos"])
                levels.append(torch.tensor([row["levels"] for row in chunk], device=device))
                targets.append(batch["target"][:, :k])
                kinds.append(batch["qtype"])
        output.append((tuple(torch.cat(parts) for parts in (features, masks, markers, levels, kinds)), torch.cat(targets)))
    return output


def forward(model, batch):
    import torch
    features, mask, markers, _, kinds = batch
    width = int(mask.sum(-1).max())
    logits, _ = model(features[:, :width], mask[:, :width], markers, torch.ones_like(markers, dtype=torch.bool), kinds)
    return logits


def measure(model, features, temperatures):
    import torch
    hits = [0, 0, 0]
    totals = [0, 0, 0]
    zero_hits = zero_total = 0
    loss = 0.0
    with torch.no_grad():
        for batch, targets in features:
            kinds = batch[-1]
            logits = forward(model, batch) / torch.tensor(temperatures, device=kinds.device)[kinds, None]
            probabilities = logits.softmax(-1)
            levels = torch.arange(logits.shape[-1], dtype=probabilities.dtype, device=logits.device)
            errors = ((probabilities - targets) @ levels).abs()
            match = torch.where(kinds == 1, errors <= .25, logits.argmax(-1) == targets.argmax(-1))
            zero = (kinds == 1) & (targets[:, 0] == 1)
            zero_hits += int((match & zero).sum())
            zero_total += int(zero.sum())
            for kind in range(3):
                mask = kinds == kind
                hits[kind] += int((match & mask).sum())
                totals[kind] += int(mask.sum())
            loss -= float((targets * logits.log_softmax(-1)).sum())
    return {"correct": hits, "total": totals, "rates": [a / b for a, b in zip(hits, totals)],
            "cross_entropy": loss / sum(totals), "level_zero": [zero_hits, zero_total]}


def fit_temperatures(model, features, retained=None, *, score_mse=False):
    import torch
    from laya.calibrate import MIN_TYPE_N, fit_one_temperature
    records = defaultdict(list)
    with torch.no_grad():
        for batch, target in features:
            for logits, y, kind in zip(forward(model, batch).cpu().numpy(), target.cpu().numpy(), batch[-1].tolist()):
                records[kind].append((logits, y))
    retained = retained or {}
    if any(len(records[kind]) < MIN_TYPE_N for kind in range(3) if kind not in retained):
        raise ValueError("each question type needs at least %d calibration examples" % MIN_TYPE_N)
    if score_mse:
        import numpy as np
        logits, gold = [torch.from_numpy(np.stack(values)) for values in zip(*records[1])]
        temperatures = torch.linspace(.5, 5, 181)
        levels = torch.arange(logits.shape[-1], dtype=logits.dtype)
        scores = (logits[None] / temperatures[:, None, None]).softmax(-1) @ levels
        losses = (scores - gold @ levels).square().mean(-1)
        retained = {**retained, 1: float(temperatures[losses.argmin()])}
    return [retained[kind] if kind in retained else fit_one_temperature(records[kind], min_n=MIN_TYPE_N)
            for kind in range(3)]


def level_zero_loss(logits, gold, original, levels, kinds, temperatures):
    import torch
    # Train the served expected score at the base temperature. The independent
    # calibration split fits the final score temperature after each epoch.
    logp = (logits / temperatures[kinds, None]).log_softmax(-1)
    anchor = original.clamp_min(1e-8).log() / temperatures[kinds, None]
    anchor = anchor.softmax(-1)
    score = kinds == 1
    target = torch.where(score[:, None], .995 * gold + .005 * anchor, .1 * gold + .9 * anchor)
    ce = -(target * logp).sum(-1)
    gold_level = (gold * levels).sum(-1)
    error = (logp.exp() * levels).sum(-1) - gold_level
    hard = 1 + 4 * (1 - (gold * anchor).sum(-1))
    weights = torch.where(score, hard * (1 + .8 * (gold_level == 0)), 4.0)
    return ((ce + score * 1.5 * error.square()) * weights).sum() / weights.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "research/evals/checkpoint-repair-training-v1.json")
    parser.add_argument("--level-zero-data", type=Path, nargs="+", help="supplemental score training and calibration banks")
    parser.add_argument("--level-zero-canonical", action="store_true",
                        help="train in presented option order and fit the score temperature to squared score error")
    parser.add_argument("--development", type=Path, nargs="+", default=[ROOT / "research/evals/qualification-v1.json",
                        ROOT / "research/evals/independent-repair-controls-v1.json"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="device for offline training")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if args.threads < 1 or args.epochs < 1 or output.exists() or output.is_relative_to(base) or not base.is_dir():
        parser.error("use a local base, a new output outside it, and positive threads and epochs")
    if args.level_zero_canonical and not args.level_zero_data:
        parser.error("--level-zero-canonical requires supplemental score training and calibration")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from laya.agent import Agent
    from laya.qualification import _artifact_identity, _file_sha, load_document, validate_corpus
    from laya.inference_common import temp_bucket

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261106)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA training requires CUDA-enabled PyTorch")
    input_paths = [args.data, Path(__file__), *args.development, *(args.level_zero_data or ())]
    inputs = {str(path.resolve()): _file_sha(path) for path in input_paths}
    names = [path.name for path in args.development]
    if len(set(names)) != len(names) or set(names) & {"training", "calibration"}:
        parser.error("development inputs need distinct filenames outside the split names")
    bank = load_document(args.data)
    splits = {part: list(training_rows(bank[part])) for part in ("training", "calibration")}
    if args.level_zero_data:
        banks = [load_document(path) for path in args.level_zero_data]
        supplement = {part: [group for bank in banks for group in bank[part]] for part in ("training", "calibration")}
        if any(group["type"] != "score" for part in ("training", "calibration") for group in supplement[part]):
            parser.error("the level-zero supplement accepts score groups only")
        splits["training"].extend(training_rows(supplement["training"]))
        splits["calibration"] = list(training_rows(supplement["calibration"]))
    dev = {path.name: [row for row in validate_corpus(load_document(path))["cases"]
                      if row["split"] == "evaluation" and row["expected"]] for path in args.development}
    seen_families, seen_states = {}, {}
    for split, rows in (splits | dev).items():
        for row in rows:
            if isinstance(row["state"], list) or any("option_order" in q for q in row["questions"].values()):
                raise ValueError("this recipe uses string/object states and default option order")
            state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
            for seen, key in ((seen_families, row["family"]), (seen_states, state)):
                if seen.setdefault(key, split) != split:
                    raise ValueError("family or state occurs in more than one split: %r" % key)

    start = time.perf_counter()
    artifacts = _artifact_identity(base)
    with safe_open(str(base / "model.safetensors"), framework="pt") as tensors:
        if any(tensors.get_slice(key).get_dtype() != "F16" for key in tensors.keys()
               if key.startswith(("head.", "scorer.", "type_emb."))):
            parser.error("this recipe expects an FP16 decision head")
    tokenizer = load_document(base / "tokenizer/tokenizer_config.json")
    if (tokenizer.get("tokenizer_class") in (None, "TokenizersBackend")
            or isinstance(tokenizer.get("extra_special_tokens"), list)):
        parser.error("normalize tokenizer metadata in a separate checkpoint copy before training")
    with Agent(str(base), device="cpu", backend="eager", performance_mode="default", decision_policy="canonical") as agent:
        if agent.parallel_options:
            raise ValueError("this recipe expects the sequential English checkpoint")
        cfg = deepcopy(agent.cfg)
        agent.model.to(args.device)
        rotate_kinds = (1,) if args.level_zero_data else (0, 1, 2)
        training = extract(agent, splits["training"], rotate_kinds=() if args.level_zero_canonical else rotate_kinds)
        print("cached training features", [len(targets) for _, targets in training], flush=True)
        calibration = extract(agent, splits["calibration"])
        development = {name: extract(agent, rows) for name, rows in dev.items()}
        temperatures = [agent.temperature_by_options.get(temp_bucket(kind, k), agent.temperature[kind])
                        for kind, k in enumerate((3, 3, 2))]
        model = agent.model
        class FrozenFeatures(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=input_ids)
        model.encoder = FrozenFeatures()
        model.act_head.requires_grad_(False)
    if _artifact_identity(base) != artifacts:
        raise RuntimeError("base checkpoint changed during feature extraction")

    before = {name: measure(model, rows, temperatures) for name, rows in development.items()}
    print("development before", before, flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    train_temperatures = torch.tensor(temperatures, device=args.device)
    retained = {kind: temperatures[kind] for kind in (0, 2)} if args.level_zero_data else None
    with torch.no_grad():
        original = [forward(model, batch).softmax(-1) for batch, _ in training]
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = [(i, ids) for i, (_, targets) in enumerate(training)
                   for ids in torch.randperm(len(targets), device=args.device).split(12)]
        for order in torch.randperm(len(batches)).tolist():
            index, ids = batches[order]
            batch, targets = training[index]
            logits = forward(model, tuple(t[ids] for t in batch))
            gold = targets[ids]
            anchor = original[index][ids]
            levels, kinds = batch[-2][ids], batch[-1][ids]
            if args.level_zero_data:
                loss = level_zero_loss(logits, gold, anchor, levels, kinds, train_temperatures)
            else:
                logp = logits.log_softmax(-1)
                target = .98 * gold + .02 * anchor
                hard = 1 + 4 * (1 - (gold * anchor).sum(-1))
                per_row = -(target * logp).sum(-1)
                ordinal_error = ((logp.exp() - gold) * levels).sum(-1)
                per_row = per_row + (kinds == 1) * ordinal_error.square()
                loss = (hard * per_row).sum() / hard.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch not in (2, 4, 6, 8, 12, 16, 24, args.epochs):
            continue
        # Score the same stored precision used by the deployed checkpoint.
        stored = deepcopy(model).half().float().eval()
        fitted = fit_temperatures(stored, calibration, retained, score_mse=args.level_zero_canonical)
        metrics = {name: measure(stored, rows, fitted) for name, rows in development.items()}
        rates = [rate for row in metrics.values() for rate in row["rates"]]
        zero_hits = sum(row["level_zero"][0] for row in metrics.values()) if args.level_zero_data else 0
        rank = (-sum(max(0, .8 - rate) for rate in rates), min(rates), zero_hits, sum(rates),
                -sum(row["cross_entropy"] for row in metrics.values()))
        row = {"epoch": epoch, "temperatures": fitted, "development": metrics}
        history.append(row)
        print(row, flush=True)
        if best is None or rank > best[0]:
            best = rank, row, stored.state_dict()

    _, selected, weights = best
    cfg["temperature"] = selected["temperatures"]
    for key in ("temperature_by_options", "lang_temperatures", "binning_map", "decision_policy"):
        cfg.pop(key, None)
    cfg["fine_tuned"] = True
    recipe = "english-level-zero-head-v1" if args.level_zero_data else "english-contrast-head-v1"
    if args.level_zero_canonical:
        recipe = "english-level-zero-canonical-v1"
    cfg["training"] = {"recipe": recipe, "selected_epoch": selected["epoch"],
                       "base_weights_sha256": artifacts["model.safetensors"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checkpoint-repair-", dir=output.parent) as tmp:
        pending = Path(tmp) / "checkpoint"
        pending.mkdir()
        for name in ("encoder", "tokenizer"):
            shutil.copytree(base / name, pending / name)
        state = load_file(str(base / "model.safetensors"))
        for key, value in weights.items():
            if key.startswith(("head.", "scorer.", "type_emb.")):
                state[key] = value.to(device="cpu", dtype=state[key].dtype).contiguous()
        save_file(state, str(pending / "model.safetensors"))
        (pending / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        report = {"base": str(base), "base_artifacts": artifacts, "candidate_artifacts": _artifact_identity(pending),
                  "input_sha256": inputs, "training_rows": len(splits["training"]),
                  "calibration_rows": len(splits["calibration"]), "epochs": args.epochs,
                  "device": args.device, "level_zero_canonical": args.level_zero_canonical,
                  "seed": 20261106, "threads": args.threads, "torch": torch.__version__,
                  "elapsed_seconds": time.perf_counter() - start, "before": before, "history": history, "selected": selected}
        if _artifact_identity(base) != artifacts or any(_file_sha(Path(p)) != digest for p, digest in inputs.items()):
            raise RuntimeError("base or training inputs changed during the run")
        (pending / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.rename(output)
    print("Saved", output, "selected epoch", selected["epoch"], flush=True)


if __name__ == "__main__":
    main()
