---
title: Evidence-grounded English checkpoint repair
description: Balanced evidence contrasts, a new sealed acceptance set and single-pass qualification.
---

# Evidence-grounded English checkpoint repair

The [level-zero follow-up](level-zero-repair-2026-11-06.md) corrected two more
level-zero scores while losing four correct higher-level answers. The follow-up
below targets the distinction between current facts, historical or hypothetical
records, and completed checks for the exact item. It also addresses routing
when a settled request and an active request mention different teams.

## Training and isolation

The new local bank has 590 labelled rows in 30 families. Current and closed
records vary independently, and their clause order changes. A severe phrase in
a closed record therefore carries no label information. For routing, a settled
request appears alongside a new action or a general inquiry. Earlier training
banks provide 259 additional examples. Calibration uses 38 separate examples.

The recipe adds a fixed-seed sample from the published WANLI training split.
Its labels distinguish support, contradiction, and unresolved evidence. The
first stage initializes the encoder with a 70% weight blend from the public
ModernBERT classifier, trains the final four encoder layers and existing head,
and scores epochs on the published development corpora. All training and
predictions ran locally. [WANLI dataset](https://huggingface.co/datasets/alisawuffles/WANLI)
and [ModernBERT classifier](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0).

In stage two, the most recent 45 published labelled cases become seen
regression training. Epoch selection uses the original rubric, older controls,
and the earlier checkpoint-repair acceptance corpus. Stage two updates the last
two encoder layers and the head. Each stage fits temperature and selects epochs
using the weight precision saved to disk. The action head weights stay frozen.
Stage one used 480 WANLI rows and selected epoch 4. Stage two used 300 WANLI
rows, repeated the 45 labelled regression rows four times, and selected epoch 2.
Each stage trained on 1,329 rows including rehearsal and repetitions.

The new acceptance set was sealed in `1c835e40` before any predictions on it.
It has 66 cases across seven distinct families, including 45 labelled examples
and 21 unlabelled boundary candidates with conflicting or incomplete evidence.
Its SHA256 is `a470c067567ed71bb504a7c7e663e76b7377743074141118271b9b92bfc7d3cd`.
The labelled scores include nine level-zero, six level-one, and six level-two
cases. Review-agent checked the rubrics and split isolation.

Qualification uses the unchanged gates. Each type needs 80% accuracy, scores
must fall within ±0.25, labels must have zero flips, probability and score drift
must be at most 0.001, and there must be at least one returned top-two gap at or
below 0.05. The new acceptance and original rubric use the full 32-profile
Torch and ONNX matrix. The other comparison fixtures use serial profiles.

## Fresh qualification

**The evidence candidate failed qualification.** All 32 fresh candidate
profiles had the same accuracy counts. The baseline ran one serial profile
per backend with the same thresholds.

| Fresh metric | Prior checkpoint | Evidence candidate |
| --- | ---: | ---: |
| Choice accuracy | 8/12 | 5/12 |
| Binary accuracy | 12/12 | 12/12 |
| Scores within ±0.25 | 7/21 | 11/21 |
| Level zero within ±0.25 | 1/9 | 4/9 |
| Level one within ±0.25 | 2/6 | 2/6 |
| Level two within ±0.25 | 4/6 | 5/6 |
| Score mean absolute error | 0.63856 | 0.36789 |

Five score cases entered tolerance and one previously correct score left it.
The candidate corrected three level-zero errors, one level-one error and one
level-two error. It lost a level-one hit on a limited floodgate monitoring
fault with working replacement coverage. It also lost both general-member-desk
answers and one certificate release routing answer. The score and choice
gates both failed.

Three cases met the measured boundary gate. One was a predeclared unlabelled
case, `canteen-allergen-release-7`, where a cropped release omits the exact lot
for a recipe used by two batches. It returned 0.5184 false and 0.4816 true,
a gap of 0.0368, in every profile. The two labelled margins were 0.0042 and
0.0228. There were zero label and operational flag flips, and maximum
probability, score and confidence drift was 0.0001. Action drift was zero.

## Earlier fixtures

The original rubric still failed strict qualification. Its 32 profiles had
3/5 choice, 5/5 binary, and 4/5 score hits, with zero measured boundaries.
The priority case with no reported impact now returns 0.0000. A previously
correct level-one priority case fell to 0.5914. Both original billing errors
remain.

The earlier checkpoint-repair acceptance corpus passed the serial gates with
10/12 choice, 12/12 binary, and 11/12 score hits. It recovered all four of that
corpus's original level-zero failures and returned two measured boundaries,
including an unlabelled margin of 0.0115. Both stages used this corpus for
epoch selection, so the pass describes a development fixture.

The older controls returned 13/15 choice, 14/15 binary, and 10/15 score
hits, with three boundaries. The prior checkpoint scored 14/15, 14/15,
and 9/15 respectively. The score gate still failed.

On the seen regression set, the candidate moved from 9/12 to 11/12 for
choice and from 10/12 to 11/12 for binary, compared with the previous
experimental checkpoint. Score hits moved from 8/21 to 16/21, including
6/9 at level zero. This rerun fell one score hit short and had no measured
boundary. Because its labelled cases were used for training, it measures
the known regressions. The fresh failures keep the candidate experimental.

## Single-pass performance

The weight comparison checked all 206 tensor names, shapes, and dtypes. The
201 changed tensors belong to the existing encoder, head, scorer, and type
embeddings. The action head and temperature buffer weights are identical.
The checkpoint configuration supplies the new fitted temperatures.

Both checkpoints ran one forward per request on Torch and ONNX. The benchmark
warmed nine labelled prompts and measured each three times with alternating
order and four CPU threads.

| Backend | Prior p50 | Candidate p50 | Prior p95 | Candidate p95 | Median paired ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager CPU | 271.7 ms | 250.7 ms | 389.2 ms | 393.5 ms | 1.076× |
| Native ONNX CPU | 236.3 ms | 234.3 ms | 302.6 ms | 271.0 ms | 0.973× |

The ratio is the median of the 27 individual ratios. The p50s are computed
for each checkpoint independently and can move in a different direction.
These small workstation samples do not establish a latency SLO. MCP Classic
and Code Mode each served two retained decisions with zero models loaded and
then observed one fresh miss and one repeat hit.

## Review and evidence

Review-agent checked the corpus split, tokenizer preflight, saved weight
precision, and verification paths. The trainer resolves all input paths
before reading and hashing them, and rejects conversation lists whose token
truncation can differ at runtime. Those two guards landed after the candidate
was frozen from the trainer in `7e6ef79d`. The supplied inputs already meet
the guards. The 327 focused qualification and ONNX tests, Ruff, compilation,
and the strict documentation and site checks passed.

Raw records under `research/results/` use the `evidence_repair_` prefix
and `_windows_20261106.json` suffix. They include `stage1`, `stage2`,
`acceptance_base`, `acceptance`, `original`, `prior_acceptance`,
`older_controls`, `seen_regression`, and `verification`.

## Reproduce

Use local artifacts for WANLI revision `61c95318fd71c55b6ba355d76253254615f387ec`
and the ModernBERT classifier revision `a51e07b524299e309dd2b88d48b0cfa2bd9ec598`.
The training records list the full commands' options, data hashes, and input
and output checkpoint hashes. Training used Torch 2.11.0+cu128 with four CPU
threads. Qualification and paired timings used the CPU runtime.

```bash
banks=(research/evals/checkpoint-repair-training-v1.json
       research/evals/level-zero-training-v1.json research/evals/level-zero-contrast-v1.json
       research/evals/evidence-repair-contrasts-v1.json)
dev=(research/evals/qualification-v1.json research/evals/independent-repair-controls-v1.json
     research/evals/checkpoint-repair-acceptance-v1.json)
regression=research/evals/level-zero-acceptance-v1.json
python research/scripts/train_evidence_repair.py --base models/english-repair-20261106 \
  --out models/english-evidence-initial-r2-20261106 --wanli /data/wanli/train.jsonl \
  --teacher /models/modernbert-classifier/model.safetensors --blend 0.7 --banks "${banks[@]}" \
  --development "${dev[@]}" "$regression" --wanli-per-label 160 --epochs 10
python research/scripts/train_evidence_repair.py --base models/english-evidence-initial-r2-20261106 \
  --out models/english-evidence-repair-r2-20261106 --wanli /data/wanli/train.jsonl \
  --banks "${banks[@]}" --development "${dev[@]}" --regression "$regression" \
  --wanli-per-label 100 --epochs 8 --tail 2 --head-lr 0.000035 --encoder-lr 0.00001
python -X utf8 scripts/export_onnx.py --model models/english-evidence-repair-r2-20261106 \
  --output models/english-evidence-repair-r2-20261106/english-evidence.onnx
python research/scripts/qualify_local.py --checkpoint models/english-evidence-repair-r2-20261106 \
  --onnx models/english-evidence-repair-r2-20261106/english-evidence.onnx --decision-policy canonical \
  --corpus research/evals/evidence-repair-acceptance-v1.json --out acceptance.json
python research/scripts/verify_checkpoint_repair.py --base models/english-repair-20261106 \
  --base-onnx models/english-repair-20261106/english-repair.onnx \
  --candidate models/english-evidence-repair-r2-20261106 \
  --onnx models/english-evidence-repair-r2-20261106/english-evidence.onnx --allow-encoder-update \
  --corpus research/evals/evidence-repair-acceptance-v1.json --out verification.json
```
