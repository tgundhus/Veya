"""Train a local single-pass checkpoint on evidence contrasts and a WANLI train sample."""
import argparse
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

from research.scripts.train_checkpoint_repair import extract, fit_temperatures, measure, training_rows


def nli_rows(path, count):
    labels = {"contradiction": 0, "neutral": 1, "entailment": 2}
    descriptions = ["The record contradicts the claim.", "The record does not establish or contradict the claim.",
                    "The record supports the claim."]
    pools = {label: [] for label in labels}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if len(row["premise"]) + len(row["hypothesis"]) <= 320:
                pools[row["gold"]].append(row)
    rng = random.Random(41203)
    for label, pool in pools.items():
        for index, row in enumerate(rng.sample(pool, count)):
            state = {"Record": row["premise"], "Claim": row["hypothesis"]}
            if index % 4 == 3:
                question = {"type": "noul", "instructions": "Does the record establish that the claim is true?"}
                expected = label == "entailment"
            elif index % 4 == 2:
                question = {"type": "choice", "instructions": "Compare the claim with the facts in the record.",
                            "criteria": dict(zip(labels, descriptions))}
                expected = label
            else:
                question = {"type": "score", "instructions": "Compare the claim with the facts in the record.",
                            "criteria": descriptions}
                expected = labels[label]
            yield {"state": state, "questions": {"evidence": question}, "expected": {"evidence": expected},
                   "family": "wanli-" + str(row["id"])}


def check_splits(splits):
    families, states = {}, {}
    for split, rows in splits.items():
        for row in rows:
            state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True).strip().casefold()
            for observed, key in ((families, row["family"]), (states, state)):
                if observed.setdefault(key, split) != split:
                    raise ValueError("family or state occurs in more than one split: %r" % key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out", "wanli"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--banks", type=Path, nargs="+", required=True)
    parser.add_argument("--development", type=Path, nargs="+", required=True)
    parser.add_argument("--regression", type=Path, help="published corpus deliberately moved into training")
    parser.add_argument("--regression-repeats", type=int, default=4)
    parser.add_argument("--teacher", type=Path, help="local ModernBERT classifier safetensors")
    parser.add_argument("--blend", type=float, default=0)
    parser.add_argument("--wanli-per-label", type=int, default=160)
    parser.add_argument("--rehearse", type=int, default=1)
    parser.add_argument("--tail", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss", choices=("soft-ce", "rlcd"), default="soft-ce")
    parser.add_argument("--head-lr", type=float, default=8e-5)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.out.resolve()
    if not base.is_dir() or output.exists() or output.is_relative_to(base):
        parser.error("use an existing local base and a new output outside it")
    if not 0 <= args.blend <= 1 or bool(args.teacher) != bool(args.blend):
        parser.error("a teacher and a positive blend fraction up to one are required together")
    if min(args.regression_repeats, args.rehearse, args.epochs) < 1 or min(args.wanli_per_label, args.tail) < 0:
        parser.error("use positive repeat and epoch counts and nonnegative sample and tail counts")
    development_names = {p.stem for p in args.development}
    if len(development_names) != len(args.development) or development_names & {"training", "calibration"}:
        parser.error("development corpora need distinct names outside the split names")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_CPU_AMP="fp32")
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from laya.agent import Agent
    from laya.qualification import _artifact_identity, _file_sha, _source_identity, load_document, validate_corpus
    from laya.train import TrainConfig, items_from_rows, train_model

    if not torch.cuda.is_available():
        parser.error("training requires a local CUDA-enabled PyTorch installation")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261106)
    device = torch.device("cuda")
    paths = [Path(__file__), Path(__file__).with_name("train_checkpoint_repair.py"),
             args.wanli, *args.banks, *args.development]
    paths.extend(path for path in (args.regression, args.teacher) if path is not None)
    inputs = {str(path.resolve()): _file_sha(path.resolve()) for path in paths}
    if len(set(inputs.values())) != len(paths):
        parser.error("training, calibration and evaluation input files must have distinct contents")
    source_hashes = _source_identity()
    base_artifacts = _artifact_identity(base)
    tokenizer = load_document(base / "tokenizer/tokenizer_config.json")
    if (tokenizer.get("tokenizer_class") in (None, "TokenizersBackend")
            or isinstance(tokenizer.get("extra_special_tokens"), list)):
        parser.error("normalize tokenizer metadata in a separate checkpoint copy before training")
    with safe_open(base / "model.safetensors", framework="pt") as stored:
        prefixes = ("encoder.", "head.", "scorer.", "type_emb.")
        if any(stored.get_slice(name).get_dtype() != "F16" for name in stored.keys() if name.startswith(prefixes)):
            parser.error("the evidence recipe requires FP16 stored encoder and decision head weights")
    banks = [load_document(path) for path in args.banks]
    training = [row for bank in banks for row in training_rows(bank["training"])]
    calibration = [row for bank in banks[1:] for row in training_rows(bank["calibration"])]
    calibration.extend(row for row in training_rows(banks[0]["calibration"])
                       if row["questions"]["decision"]["type"] != "score")

    def evaluation(path):
        return [row for row in validate_corpus(load_document(path))["cases"] if row["split"] == "evaluation"]

    dev = {path.stem: evaluation(path) for path in args.development}
    regression = evaluation(args.regression) if args.regression else []
    natural_language = list(nli_rows(args.wanli, args.wanli_per_label))
    splits = {"training": training + regression + natural_language, "calibration": calibration, **dev}
    check_splits(splits)
    if any("option_order" in question for split in splits.values() for row in split
           for question in row["questions"].values()):
        parser.error("the evidence recipe requires the default option order")
    regression = [row for row in regression if row["expected"]]
    rows = training * args.rehearse + natural_language + regression * args.regression_repeats
    dev = {name: [row for row in cases if row["expected"]] for name, cases in dev.items()}

    class FrozenFeatures(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids)

    history, best = [], None
    with Agent(str(base), device="cpu", backend="eager", performance_mode="default", decision_policy="canonical") as agent:
        if agent.parallel_options:
            raise ValueError("this recipe requires the sequential layout")
        model = agent.model.to(device)
        if args.tail > len(model.encoder.layers):
            parser.error("encoder tail exceeds the checkpoint layer count")
        model.encoder.requires_grad_(False)
        if args.tail:
            for layer in model.encoder.layers[-args.tail:]:
                layer.requires_grad_(True)
        model.act_head.requires_grad_(False)
        changed = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if args.teacher:
            with safe_open(args.teacher, framework="pt") as teacher, torch.no_grad():
                for name, parameter in model.encoder.named_parameters():
                    incoming = teacher.get_tensor("model." + name)
                    if incoming.shape != parameter.shape:
                        raise ValueError("teacher encoder tensor has incompatible shape: %s" % name)
                    parameter.lerp_(incoming.to(device=device, dtype=parameter.dtype), args.blend)
                    if not parameter.requires_grad:
                        parameter.copy_(parameter.half())
                    changed.add("encoder." + name)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

        max_len, head_len = (agent.cfg[key] for key in ("max_len", "head_max_len"))
        items, skipped = items_from_rows(agent.tok, rows, max_len, head_len)
        if skipped or len(items) != sum(len(row["expected"]) for row in rows):
            raise ValueError("could not encode every training answer: %r" % skipped)

        @torch.no_grad()
        def evaluate(epoch, _):
            nonlocal best
            model.eval()
            # Rank and calibrate the serialized weights while keeping the live
            # optimizer parameters at full precision between epochs.
            live = [parameter.detach().clone() for parameter in trainable]
            try:
                for parameter in trainable:
                    parameter.copy_(parameter.half())
                features = {name: extract(agent, cases) for name, cases in dev.items()}
                calibration_features = extract(agent, calibration)
                encoder = model.encoder
                try:
                    model.encoder = FrozenFeatures()
                    temperatures = fit_temperatures(model, calibration_features)
                    metrics = {name: measure(model, feat, temperatures) for name, feat in features.items()}
                finally:
                    model.encoder = encoder
                rates = [rate for item in metrics.values() for rate in item["rates"]]
                rank = (-sum(max(0, .8 - rate) for rate in rates), min(rates), sum(rates),
                        -sum(item["cross_entropy"] for item in metrics.values()))
                row = {"epoch": epoch + 1, "temperatures": temperatures, "development": metrics}
                history.append(row)
                print(row, flush=True)
                if best is None or rank > best[0]:
                    weights = {name: value.detach().to(device="cpu", dtype=torch.float16)
                               for name, value in model.state_dict().items() if name in changed}
                    best = rank, row, weights
            finally:
                for parameter, value in zip(trainable, live):
                    parameter.copy_(value)
                model.train()

        print("training", len(items), "regression", len(regression), "calibration", len(calibration), flush=True)
        config = TrainConfig(epochs=args.epochs, micro_batch=20, grad_accum=2, encoder_lr=args.encoder_lr,
                             head_lr=args.head_lr, loss=args.loss, seed=20261106, amp=True,
                             gradient_checkpointing=False)
        train_model(model, agent.tok, items, config, device, max_len, head_len, on_epoch_end=evaluate)
        _, selected, weights = best
        cfg = dict(agent.cfg)

    cfg["temperature"] = selected["temperatures"]
    for key in ("temperature_by_options", "lang_temperatures", "binning_map", "decision_policy"):
        cfg.pop(key, None)
    cfg["training"] = {"recipe": "english-evidence-repair-v1", "selected_epoch": selected["epoch"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evidence-repair-", dir=output.parent) as temporary:
        pending = Path(temporary) / "checkpoint"
        pending.mkdir()
        for name in ("encoder", "tokenizer"):
            shutil.copytree(base / name, pending / name)
        state = load_file(base / "model.safetensors")
        for name, value in weights.items():
            state[name] = value.to(state[name].dtype).contiguous()
        save_file(state, pending / "model.safetensors")
        (pending / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        if (_artifact_identity(base) != base_artifacts or _source_identity() != source_hashes or
                any(_file_sha(Path(path)) != digest for path, digest in inputs.items())):
            raise RuntimeError("checkpoint, inputs or runtime changed during training")
        report = {"selected": selected, "history": history, "training_rows": len(rows),
                  "regression_rows": len(regression), "calibration_rows": len(calibration),
                  "wanli_rows": len(natural_language), "torch": torch.__version__, "seed": config.seed,
                  "args": {key: [str(p) for p in value] if isinstance(value, list) else
                           str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "changed_tensor_names": sorted(changed), "input_sha256": inputs,
                  "base_artifacts": base_artifacts, "candidate_artifacts": _artifact_identity(pending),
                  "source_sha256": source_hashes}
        (pending / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.rename(output)
    print("Saved", output, "selected epoch", selected["epoch"], flush=True)


if __name__ == "__main__":
    main()
