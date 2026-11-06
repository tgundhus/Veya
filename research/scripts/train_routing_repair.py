"""Train only the choice adapters of an existing local decision checkpoint."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.scripts.train_checkpoint_repair import training_rows
from research.scripts.train_evidence_repair import check_splits


def choice_rows(rows):
    for row in rows:
        for qid, expected in row["expected"].items():
            if row["questions"][qid]["type"] == "choice":
                yield {**row, "questions": {qid: row["questions"][qid]}, "expected": {qid: expected}}


def clinc_rows(data, domains):
    """Build three-option questions from CLINC train and oos_train only."""
    rng = random.Random(4421)
    neighbors = {name: [other for other in group if other != name] for group in domains.values() for name in group}
    queries = defaultdict(list)
    for state, label in data["train"]:
        queries[label].append(state)

    def case(state, family, options, target):
        rng.shuffle(options)
        criteria = {name: ("General inquiries or the request does not match either listed team."
                           if name == "front_desk" else name.replace("_", " ")) for name in options}
        question = {"type": "choice", "criteria": criteria,
                    "instructions": "Route the current request to the team that owns it. Use the front desk "
                                    "when the listed teams do not cover it or the request is unspecified."}
        return {"family": "clinc-" + family, "state": state, "questions": {"route": question},
                "expected": {"route": target}}

    for label, states in queries.items():
        related = neighbors[label]
        for index, state in enumerate(rng.sample(states, 18)):
            negative = rng.choice(related)
            options, target = [label, negative, "front_desk"], label
            if index % 6 == 0:
                options, target = [*rng.sample(related, 2), "front_desk"], "front_desk"
            if index % 3 == 1:
                state = "My older request was settled: %s. My current question is: %s" % (rng.choice(queries[negative]), state)
            if index % 9 == 2:
                state = ("I had asked: %s. That is settled. I'll give my reference first and explain the new request later."
                         % rng.choice(queries[negative]))
                target = "front_desk"
            yield case(state, label, options, target)
    for state, _ in data["oos_train"]:
        yield case(state, "oos", [*rng.sample(list(queries), 2), "front_desk"], "front_desk")


def capture(agent, rows, *, rotate=False, compact=False):
    import torch
    from torch.nn.functional import pad
    from laya.common import collate_items
    from laya.train import encode_item, items_from_rows

    max_len, head_len = (agent.cfg[key] for key in ("max_len", "head_max_len"))
    items, skipped = items_from_rows(agent.tok, rows, max_len, head_len)
    if skipped or len(items) != len(rows) or not items or any(item["k"] != 3 for item in items):
        raise ValueError("the routing recipe needs three-option choice rows: %r" % skipped)
    orders = (None, [1, 2, 0], [2, 0, 1]) if rotate else (None,)
    encoded = [encode_item(agent.tok, item, max_len, head_len, order) for item in items for order in orders]
    width = max(len(item["ids"]) for item in encoded)
    features, masks, markers, logits, targets = [], [], [], [], []

    def remember(_, inputs, output):
        hidden = output.last_hidden_state.detach()
        if compact:
            hidden = hidden.half()
        features.append(pad(hidden, (0, 0, 0, width - hidden.shape[1])))

    device = next(agent.model.parameters()).device
    with torch.no_grad(), agent.model.encoder.register_forward_hook(remember):
        for start in range(0, len(encoded), 32):
            batch = collate_items([encoded[start:start + 32]], agent.tok.pad_token_id)
            args = [batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            result, _ = agent.model(*args)
            masks.append(pad(args[1], (0, width - args[1].shape[1])))
            markers.append(args[2])
            logits.append(result)
            targets.append(batch["target"].to(device))
    return tuple(torch.cat(values) for values in (features, masks, markers, logits, targets))


def forward(model, features):
    import torch
    hidden, mask, markers = features
    width = int(mask.sum(-1).max())
    return model(hidden[:, :width].float(), mask[:, :width], markers, torch.ones_like(markers, dtype=torch.bool),
                 torch.zeros(len(mask), device=mask.device, dtype=torch.long))[0]


def main():
    def local_path(value):
        return Path(value).resolve()

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out", "clinc", "domains", "regression"):
        parser.add_argument("--" + name, type=local_path, required=True)
    parser.add_argument("--banks", nargs="+", type=local_path, required=True)
    parser.add_argument("--development", nargs="+", type=local_path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if not args.base.is_dir() or args.out.exists() or args.out.is_relative_to(args.base) or args.epochs < 1:
        parser.error("use a local base, new output outside it, and a positive epoch count")
    reserved = {"training", "calibration", "seen_regression"}
    names = {path.stem for path in args.development}
    if names & reserved or len(names) != len(args.development):
        parser.error("development sets need distinct names outside the reserved split names")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
    import torch
    from safetensors.torch import load_file, save_file
    from laya.agent import Agent
    from laya.calibrate import MIN_TYPE_N, fit_one_temperature
    from laya.common import DecisionModel
    from laya.qualification import _artifact_identity, _file_sha, _source_identity, load_document, validate_corpus

    if not torch.cuda.is_available():
        parser.error("this research trainer requires local CUDA-enabled PyTorch")
    torch.manual_seed(20261106)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sources = _source_identity()
    scripts = [Path(__file__), *[Path(__file__).with_name(name) for name in
                               ("train_checkpoint_repair.py", "train_evidence_repair.py")]]
    inputs = {str(path): _file_sha(path) for path in [*scripts, args.clinc, args.domains, args.regression,
                                                     *args.banks, *args.development]}
    artifacts = _artifact_identity(args.base)
    tokenizer = load_document(args.base / "tokenizer/tokenizer_config.json")
    if (tokenizer.get("tokenizer_class") in (None, "TokenizersBackend")
            or isinstance(tokenizer.get("extra_special_tokens"), list)):
        parser.error("normalize tokenizer metadata in a separate checkpoint copy before training")

    def cases(path):
        return [case for case in validate_corpus(load_document(path))["cases"] if case["split"] == "evaluation"]

    banks = [load_document(path) for path in args.banks]
    local = list(choice_rows(row for bank in banks for row in training_rows(bank["training"])))
    calibration = list(choice_rows(row for bank in banks for row in training_rows(bank["calibration"])))
    clinc = list(clinc_rows(load_document(args.clinc), load_document(args.domains)))
    regression = cases(args.regression)
    dev = {path.stem: cases(path) for path in args.development}
    check_splits({"training": local + clinc + regression, "calibration": calibration, **dev})
    reg_choices = list(choice_rows(regression))
    local.extend(reg_choices * 5)
    dev = {name: list(choice_rows(rows)) for name, rows in dev.items()}
    dev["seen_regression"] = reg_choices

    with Agent(str(args.base), device="cpu", backend="eager", performance_mode="default", decision_policy="canonical") as agent:
        if (agent.cfg.get("choice_adapter_width") or agent.parallel_options or agent.model.head is None
                or agent.cfg.get("temperature_by_options") or agent.cfg.get("lang_temperatures")):
            parser.error("use an unadapted sequential checkpoint with a decision head and scalar type temperatures")
        agent.model.cuda().eval()
        training = (capture(agent, clinc, compact=True), capture(agent, local, rotate=True, compact=True))
        development = {name: capture(agent, rows) for name, rows in dev.items()}
        calibration_features = capture(agent, calibration)
        cfg = deepcopy(agent.cfg)
        old_head = {key: value for key, value in agent.model.state_dict().items() if not key.startswith("encoder.")}
        hidden_size = agent.model.encoder.config.hidden_size

        class Features(torch.nn.Module):
            config = SimpleNamespace(hidden_size=hidden_size)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=input_ids)

        model = DecisionModel(Features(), cfg["head_layers"], len(cfg.get("act_costs", {})) + 1,
                              choice_adapter_width=64).cuda().eval()
        missing, unexpected = model.load_state_dict(old_head, strict=False)
        assert missing and all(name.startswith("choice_adapters.") for name in missing) and not unexpected
    del old_head
    model.requires_grad_(False)
    adapters = model.choice_adapters.requires_grad_(True)
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=3e-4, weight_decay=.015)
    best, history = None, []

    @torch.no_grad()
    def evaluate(epoch):
        nonlocal best
        stored = deepcopy(adapters)
        model.choice_adapters = stored
        try:
            for strength in (.5, .75, 1.):
                stored.load_state_dict(adapters.state_dict())
                for block in stored:
                    for parameter in block[-1].parameters():
                        parameter.mul_(strength)
                stored.half().float()
                metrics = {}
                for name, data in development.items():
                    predicted = forward(model, data[:3])
                    target = data[-1]
                    metrics[name] = {"correct": int((predicted.argmax(-1) == target.argmax(-1)).sum()),
                                     "total": len(target), "loss": float(-(target * predicted.log_softmax(-1)).sum(-1).mean())}
                rates = [row["correct"] / row["total"] for row in metrics.values()]
                rank = (-sum(max(0, .8 - rate) for rate in rates), min(rates), sum(rates),
                        -sum(row["loss"] for row in metrics.values()))
                row = {"epoch": epoch, "strength": strength, "development": metrics}
                history.append(row)
                print(row, flush=True)
                if best is None or rank > best[0]:
                    logits = forward(model, calibration_features[:3]).cpu().numpy()
                    labels = calibration_features[-1].cpu().numpy()
                    temperature = fit_one_temperature(list(zip(logits, labels)), min_n=MIN_TYPE_N)
                    best = rank, {**row, "temperature": temperature}, {
                        "choice_adapters." + name: value.half().cpu() for name, value in stored.state_dict().items()}
        finally:
            model.choice_adapters = adapters

    for epoch in range(1, args.epochs + 1):
        steps = [(i, idx % len(data[-1])) for i, data in enumerate(training)
                 for idx in torch.randperm(len(data[-1]) * (3 if i else 1), device="cuda").split(48)]
        for order in torch.randperm(len(steps)).tolist():
            index, ids = steps[order]
            h, mask, markers, base, target = (value[ids] for value in training[index])
            logits = forward(model, (h, mask, markers))
            correct = base.argmax(-1) == target.argmax(-1)
            anchor = torch.where(correct, .15, .01)[:, None]
            mixed = (1 - anchor) * target + anchor * base.softmax(-1)
            loss = -((mixed * logits.log_softmax(-1)).sum(-1) * (2 - correct.float())).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), 2.)
            optimizer.step()
        if epoch in (1, 2, 3, 4, 6, 8, 12, args.epochs):
            evaluate(epoch)

    _, selected, weights = best
    cfg["choice_adapter_width"] = 64
    cfg["temperature"][0] = selected["temperature"]
    cfg["training"] = {"recipe": "english-choice-adapter-v1", "selected_epoch": selected["epoch"],
                       "base_weights_sha256": artifacts["model.safetensors"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="routing-repair-", dir=args.out.parent) as temporary:
        pending = Path(temporary) / "checkpoint"
        pending.mkdir()
        for name in ("encoder", "tokenizer"):
            shutil.copytree(args.base / name, pending / name)
        state = load_file(args.base / "model.safetensors")
        save_file({**state, **weights}, pending / "model.safetensors")
        (pending / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        if (_artifact_identity(args.base) != artifacts or _source_identity() != sources or
                any(_file_sha(Path(name)) != digest for name, digest in inputs.items())):
            raise RuntimeError("checkpoint, input or runtime source changed during training")
        report = {"selected": selected, "history": history, "clinc_rows": len(clinc), "local_rows": len(local),
                  "calibration_rows": len(calibration), "regression_rows": len(reg_choices), "epochs": args.epochs,
                  "torch": torch.__version__, "seed": 20261106, "input_sha256": inputs, "source_sha256": sources,
                  "base_artifacts": artifacts, "candidate_artifacts": _artifact_identity(pending)}
        (pending / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.rename(args.out)
    print("Saved", args.out, selected, flush=True)


if __name__ == "__main__":
    main()
