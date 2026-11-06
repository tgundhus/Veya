"""Fit score adapters on local ordinal labels and ambiguous evidence, with the base frozen."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.scripts.train_checkpoint_repair import extract, forward, training_rows
from research.scripts.train_evidence_repair import check_splits


def scores(rows, *, boundaries=False):
    for row in rows:
        for qid, question in row["questions"].items():
            if question["type"] != "score" or (qid not in row["expected"] and not boundaries):
                continue
            # Feature extraction needs a target to encode the row. An unlabelled
            # diagnostic target is never used to fit weights or count accuracy.
            yield {**row, "questions": {qid: question}, "expected": {qid: row["expected"].get(qid, 0)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out", "bank"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    for name in ("rehearsal", "regression", "development"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=32)
    args = parser.parse_args()
    if not args.base.is_dir() or args.out.exists() or args.out.is_relative_to(args.base) or args.epochs < 1:
        parser.error("use a local base, a new output outside it, and positive epochs")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
    import torch
    from safetensors.torch import load_file, save_file
    from laya.agent import Agent
    from laya.common import DecisionModel
    from laya.qualification import _artifact_identity, _file_sha, _source_identity, load_document, validate_corpus

    if not torch.cuda.is_available():
        parser.error("this research trainer requires local CUDA-enabled PyTorch")
    torch.manual_seed(20261107)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    paths = [Path(__file__), Path(__file__).with_name("train_checkpoint_repair.py"),
             Path(__file__).with_name("train_evidence_repair.py"), args.bank,
             *args.rehearsal, *args.regression, *args.development]
    inputs = {str(path): _file_sha(path) for path in paths}
    eval_paths = [*args.regression, *args.development]
    if len(set(inputs.values())) != len(paths) or len({p.stem for p in eval_paths}) != len(eval_paths):
        parser.error("all input files must have distinct contents and evaluation names")
    artifacts, source = _artifact_identity(args.base), _source_identity()
    bank = load_document(args.bank)
    local = list(scores(training_rows(bank["training"])))
    calibration = list(scores(training_rows(bank["calibration"])))
    rehearsal = list(scores(row for path in args.rehearsal
                            for row in training_rows(load_document(path)["training"])))
    cal_gold = {row["expected"]["decision"] for row in calibration}
    if not local or not cal_gold.intersection({.5, 1.5}) or not cal_gold.intersection({0, 1, 2}):
        parser.error("the bank must supply training labels and both hard and adjacent-level calibration targets")

    def cases(path):
        return [row for row in validate_corpus(load_document(path))["cases"] if row["split"] == "evaluation"]

    seen = {path.stem: list(scores(cases(path))) for path in args.regression}
    development = {path.stem: cases(path) for path in args.development}
    regression = [row for rows in seen.values() for row in rows]
    if not all(seen.values()) or not regression:
        parser.error("each seen regression corpus must supply labelled score cases")
    check_splits({"training": local + rehearsal + regression, "calibration": calibration,
                  **{"dev:" + name: rows for name, rows in development.items()}})

    class Features(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=width)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids)

    def capture(agent, rows, *, rotate=False):
        (batch, target), = extract(agent, rows, rotate_kinds=(1,) if rotate else ())
        hidden, *other = batch
        return (hidden.half(), *other), target

    def join_rotations(training, rotated, alternate):
        from torch.nn.functional import pad
        hidden, mask, *columns = training
        extra_hidden, extra_mask, *extra_columns = rotated
        width = max(mask.shape[1], extra_mask.shape[1])
        old_pad, new_pad = width - mask.shape[1], width - extra_mask.shape[1]
        return (torch.cat((pad(hidden, (0, 0, 0, old_pad)), pad(extra_hidden[alternate], (0, 0, 0, new_pad)))),
                torch.cat((pad(mask, (0, old_pad)), pad(extra_mask[alternate], (0, new_pad)))),
                *(torch.cat((a, b[alternate])) for a, b in zip(columns, extra_columns)))

    with Agent(str(args.base), device="cpu", backend="eager", performance_mode="default",
               decision_policy="canonical") as agent:
        cfg = deepcopy(agent.cfg)
        if (not cfg.get("choice_adapter_width") or agent.parallel_options
                or cfg.get("temperature_by_options") or cfg.get("lang_temperatures")):
            parser.error("use a sequential choice-adapted checkpoint with scalar type temperatures")
        continuing = bool(cfg.get("score_adapter_width"))
        width = cfg.get("score_adapter_width") or 64
        agent.model.cuda().eval()
        training, targets = capture(agent, local + rehearsal + regression)
        rotated_start = len(targets)
        if continuing:
            rotated, rotated_targets = capture(agent, local + regression, rotate=True)
            alternate = (rotated[-2][:, 0] != 0)
            training = join_rotations(training, rotated, alternate)
            targets = torch.cat((targets, rotated_targets[alternate]), 0)
            del rotated, rotated_targets
        cal, cal_targets = capture(agent, calibration)
        dev = {name: capture(agent, list(scores(rows))) for name, rows in development.items()}
        dev.update({name: capture(agent, rows) for name, rows in seen.items()})
        bounds, boundary_ids = {}, {}
        for name, rows in development.items():
            candidates = list(scores((row for row in rows if not row["expected"]), boundaries=True))
            if candidates:
                bounds[name] = capture(agent, candidates)[0]
                boundary_ids[name] = [(row["id"], next(iter(row["questions"]))) for row in candidates]
        model = DecisionModel(Features(agent.model.encoder.config.hidden_size), cfg["head_layers"],
                              len(cfg.get("act_costs", {})) + 1, choice_adapter_width=cfg["choice_adapter_width"],
                              score_adapter_width=width).cuda().eval()
        head = {key: value for key, value in agent.model.state_dict().items() if not key.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(head, strict=False)
        expected_missing = set() if continuing else {"score_adapters." + name for name in model.score_adapters.state_dict()}
        assert set(missing) == expected_missing and not unexpected
        del head
    del agent
    model.requires_grad_(False)
    adapters = model.score_adapters.requires_grad_(True)
    initial = {name: value.detach().clone() for name, value in adapters.state_dict().items()} if continuing else None
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=8e-5 if continuing else 4e-4, weight_decay=.015)
    temperatures = torch.linspace(.5, 4., 141, device="cuda")
    levels = torch.arange(3., device="cuda")
    cal_score = cal_targets @ levels
    cal_ambiguous = cal_targets.max(-1).values < 1
    base_temperature = cfg["temperature"][1]

    def predict(batch):
        hidden, *other = batch
        return forward(model, (hidden.float(), *other))

    @torch.no_grad()
    def base_predictions():
        return torch.cat([predict(tuple(value[start:start + 48] for value in training))
                          for start in range(0, len(targets), 48)]).div(base_temperature).softmax(-1)

    original = base_predictions()
    hard = (targets.max(-1).values == 1)
    midpoint = targets.max(-1).values == .5
    semantic_levels = training[-2].float()
    correct = ((original - targets) * semantic_levels).sum(-1).abs() <= .25
    weights = torch.ones_like(correct, dtype=torch.float32)
    weights[:len(local)] = 2
    weights[rotated_start - len(regression):rotated_start] = 3
    weights[rotated_start:] = .35
    weights *= torch.where(hard, torch.where(correct, 1., 2.), 3.)
    anchor = torch.where(hard & correct, .35 if continuing else .08, .0)[:, None]
    mixed = (1 - anchor) * targets + anchor * original
    mixed = torch.where(hard[:, None], .99 * mixed + .01 / len(levels), mixed)
    best, history = None, []

    @torch.no_grad()
    def evaluate(epoch):
        nonlocal best
        stored = deepcopy(adapters)
        model.score_adapters = stored
        try:
            for strength in (.65, .85, 1.):
                current = adapters.state_dict()
                if continuing:
                    stored.load_state_dict({name: initial[name].lerp(value, strength)
                                            for name, value in current.items()})
                else:
                    stored.load_state_dict(current)
                    for block in stored:
                        for parameter in block[-1].parameters():
                            parameter.mul_(strength)
                stored.half().float()
                logits = predict(cal)
                logp = (logits[None] / temperatures[:, None, None]).log_softmax(-1)
                errors = (logp.exp() @ levels - cal_score).abs()
                # Calibrate the served expectation and the distribution over plausible levels.
                cross_entropy = -(logp * cal_targets).sum(-1)
                loss = errors.square().mean(-1) + cross_entropy.mean(-1)
                if continuing:
                    # Choose among temperatures that preserve the most hard
                    # calibration scores, then use proper loss to break ties.
                    hits = (errors[:, ~cal_ambiguous] <= .25).sum(-1)
                    loss = loss.masked_fill(hits != hits.max(), torch.inf)
                temperature = float(temperatures[loss.argmin()])
                probability = (logits / temperature).softmax(-1)
                top = probability[cal_ambiguous].topk(2).values
                cal_gaps = (top[:, 0] - top[:, 1]).tolist()
                cal_correct = int(((probability @ levels - cal_score).abs()[~cal_ambiguous] <= .25).sum())
                cal_total = int((~cal_ambiguous).sum())
                metrics = {}
                for name, (batch, gold) in dev.items():
                    errors = (((predict(batch) / temperature).softmax(-1) - gold) @ levels).abs()
                    metrics[name] = {"correct": int((errors <= .25).sum()), "total": len(gold),
                                     "mae": float(errors.mean())}
                margins, boundary_top = {}, {}
                for name, batch in bounds.items():
                    top = (predict(batch) / temperature).softmax(-1).topk(2)
                    margins[name] = (top.values[:, 0] - top.values[:, 1]).tolist()
                    boundary_top[name] = [{"id": cid, "qid": qid, "levels": indices}
                                          for (cid, qid), indices in zip(boundary_ids[name], top.indices.tolist())]
                rates = [row["correct"] / row["total"] for row in metrics.values()] + [cal_correct / cal_total]
                coverage = sum(min(gaps) <= .05 for gaps in margins.values())
                rank = (-sum(max(0, .8 - rate) for rate in rates), coverage, min(rates), sum(rates),
                        -sum(row["mae"] for row in metrics.values()), -sum(cal_gaps))
                record = {"epoch": epoch, "strength": strength, "score_temperature": temperature,
                          "development": {name: metrics[name] for name in development},
                          "seen_regression": {name: metrics[name] for name in seen},
                          "development_boundary_gaps": margins,
                          "development_boundary_top_levels": boundary_top,
                          "calibration_hard_scores": [cal_correct, cal_total],
                          "calibration_boundary_gaps": cal_gaps}
                history.append(record)
                print(record, flush=True)
                if best is None or rank > best[0]:
                    best = rank, record, {"score_adapters." + name: tensor.half().cpu()
                                          for name, tensor in stored.state_dict().items()}
        finally:
            model.score_adapters = adapters

    print("examples:", len(local), len(rehearsal), len(regression), len(calibration), flush=True)
    for epoch in range(1, args.epochs + 1):
        indices = torch.multinomial(weights, len(targets) * 2, replacement=True)
        for idx in indices.split(48):
            logits = predict(tuple(value[idx] for value in training))
            logp = (logits / base_temperature if continuing else logits).log_softmax(-1)
            probability = logp.exp()
            error = ((probability - targets[idx]) * semantic_levels[idx]).sum(-1).square()
            if continuing:
                # Give both plausible adjacent levels comparable mass without
                # rewarding uncertainty on examples with a verified level.
                adjacent = targets[idx] > 0
                pair_gap = probability.masked_fill(~adjacent, 0).amax(-1) - probability.masked_fill(~adjacent, 1).amin(-1)
                error = error + torch.where(midpoint[idx], pair_gap.square(), 0)
            loss = (-(mixed[idx] * logp).sum(-1) + error).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), 2.)
            optimizer.step()
        if epoch in (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, args.epochs):
            evaluate(epoch)

    _, selected, added = best
    cfg["score_adapter_width"] = width
    cfg["temperature"][1] = selected["score_temperature"]
    cfg["training"] = {"recipe": "english-score-continuation-v1" if continuing else "english-score-boundary-v1",
                       "selected_epoch": selected["epoch"],
                       "base_weights_sha256": artifacts["model.safetensors"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="score-boundary-", dir=args.out.parent) as temporary:
        pending = Path(temporary) / "checkpoint"
        pending.mkdir()
        for name in ("encoder", "tokenizer"):
            shutil.copytree(args.base / name, pending / name)
        save_file({**load_file(args.base / "model.safetensors"), **added}, pending / "model.safetensors")
        (pending / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        if (_artifact_identity(args.base) != artifacts or _source_identity() != source
                or any(_file_sha(Path(name)) != digest for name, digest in inputs.items())):
            raise RuntimeError("checkpoint, input or runtime source changed during training")
        report = {"selected": selected, "history": history, "local_rows": len(local), "rehearsal_rows": len(rehearsal),
                  "regression_rows": len(regression), "calibration_rows": len(calibration), "epochs": args.epochs,
                  "rotated_rows": len(targets) - rotated_start,
                  "input_roles": {"bank": str(args.bank), "rehearsal": [str(p) for p in args.rehearsal],
                                  "seen_regression": [str(p) for p in args.regression],
                                  "development": [str(p) for p in args.development]},
                  "torch": torch.__version__, "seed": 20261107, "input_sha256": inputs, "source_sha256": source,
                  "base_artifacts": artifacts, "candidate_artifacts": _artifact_identity(pending)}
        (pending / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.rename(args.out)
    print("Saved", args.out, selected, flush=True)


if __name__ == "__main__":
    main()
