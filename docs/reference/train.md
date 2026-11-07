# Fine-tuning

`veya.train` is imported on its own (`from veya.train import finetune, TrainConfig`); it is not
re-exported from the top-level package. `veya-train` is its command line. The
[fine-tuning guide](../finetune.md#fine-tune-with-veya-train) shows both in use.

## Running a fine-tune

::: veya.train.TrainConfig

::: veya.train.finetune

::: veya.train.dry_run

## The two halves of `finetune`

::: veya.train.train_model

::: veya.train.calibration_records

::: veya.train.calibration_report

## Reading data

::: veya.train.read_data

::: veya.train.rows_from_csv

## Building training items

::: veya.train.items_from_rows

::: veya.train.target_from_gold

::: veya.train.target_from_expected

::: veya.train.to_internal

::: veya.train.make_item

::: veya.train.encode_item

::: veya.train.draw_option_order

::: veya.train.split_calibration

## Losses

::: veya.train.soft_ce_loss

::: veya.train.rlcd_loss

## Checkpoints

::: veya.train.resolve_checkpoint_dir

::: veya.train.load_checkpoint

::: veya.train.save_checkpoint
