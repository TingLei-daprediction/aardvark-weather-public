# RTMA-OK hourly-background experiment

## Purpose

Test whether an hour-matched URMA first guess improves the encoder relative to persisting each
day's 00 UTC first guess. The model architecture and 24-channel input layout are unchanged.

## Modes and files

`--background_mode daily_00z` is the backward-compatible default and reads
`urma/background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap` with one frame per day.

`--background_mode hourly` requires `--time_freq 1H` and reads
`urma/background_hourly_rtma_ok_sfc_1_<YYYY>-<MM>.memmap` with 24 frames per day. The hourly
background, observation, and target use the same `month_frame()` index.

Hourly raw and normalized products have separate names, so preparation cannot overwrite the
daily product. Backgrounds are normalized with the target mean/std, not their own statistics.

## Mandatory pre-training gate

The complete dependency chain can be submitted from a login node with:

```bash
bash scripts/run_build_urma_ok_hourly_background.sh
```

By default this includes the 2022-only normalization job. If those factors and the daily control
backgrounds have already been recomputed and verified, use
`RECOMPUTE_TRAINING_NORMS=0 bash scripts/run_build_urma_ok_hourly_background.sh` to submit only
the hourly array, hourly normalization, and diagnostics.

First recompute all loader-facing normalization factors from 2022 and re-normalize the daily
background arm:

```bash
sbatch scripts/run_compute_rtma_ok_2022_norms.sbatch
```

Then create all 24 raw hourly months as a four-way SLURM array and submit normalization only after
every array task succeeds:

```bash
prep_job=$(sbatch --parsable scripts/run_prep_urma_ok_hourly_background.sbatch)
sbatch --dependency=afterok:${prep_job} \
  scripts/run_normalize_urma_ok_hourly_background.sbatch
```

Normalization checks all requested raw months before writing anything, treats a missing month as
an error, and processes 24 frames at a time rather than materializing a full month in memory.

Then run the CPU-only quality and leakage diagnostic:

```bash
sbatch scripts/run_diagnose_urma_ok_hourly_background.sbatch
```

The diagnostic writes separate 2022 and 2023 reports. It reports bias and RMSE by variable and UTC
hour for both `GES[H]-analysis[H]` and `GES[H]-analysis[H-1]`, plus their RMSE ratio. It fails on
an empty channel-hour bin or if any individual channel-hour has near-zero normalized RMSE or a
near-zero current/previous RMSE ratio. Review both CSVs before allocating GPUs. The 2023
same-valid-time background RMSE is the validation baseline the trained model must beat.

## Fixed split and A/B protocol

Training is all timestamps in 2022; validation is all timestamps in 2023. The exact months are
recorded in `training/rtma_ok_train_months_2022.txt` and
`training/rtma_ok_val_months_2023.txt`. Static observation norms and target norms must be computed
from 2022 only. No 2023 sample may enter normalization or fitting.

During training, validation retains every seventh timestamp of 2023 (`--assim_val_stride 7`,
1,252 samples). Because 7 is coprime with 24, this selection rotates through all UTC hours
instead of aliasing onto fixed synoptic hours. Run the full hour-by-hour 2023 evaluation once
from each selected final checkpoint.

Launch both arms from fresh initialization with the same script:

```bash
sbatch --export=ALL,BACKGROUND_MODE=daily_00z \
  training/new-bsize_2-train-12month-selected-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static.sh

sbatch --export=ALL,BACKGROUND_MODE=hourly \
  training/new-bsize_2-train-12month-selected-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static.sh
```

The output directories include the background mode and therefore cannot collide. Do not place an
old checkpoint in either new output directory. Both arms must use the same optimizer settings,
training samples, validation samples, normalization, number of optimizer steps, grid, and fixed
`--seed 202209`. This controls initialization and rank-local random streams but does not promise
bit-for-bit CUDA determinism.

The training script refuses to start a fresh run if its mode-specific output directory exists.
Each completed epoch atomically updates `checkpoint_last` with model, optimizer, scheduler, epoch,
and best-loss state. Resume an interrupted arm explicitly:

```bash
sbatch --export=ALL,BACKGROUND_MODE=hourly,RESUME_CHECKPOINT=/path/to/hourly/output/checkpoint_last \
  training/new-bsize_2-train-12month-selected-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static.sh
```

The resume checkpoint must be inside the expected mode-specific output directory. Training
continues at `checkpoint_epoch + 1` toward the configured total of 900 epochs.

For an hourly-only run derived directly from the earlier three-month Grid-B script, submit:

```bash
sbatch training/new-bsize_2-train-2022-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static-hourly-background.sh
```

This entry point fixes `--background_mode hourly`, trains on all of 2022, validates on the
stride-7 2023 holdout, performs the file preflight check, and uses its own output directory and
master port. Resume it by supplying its mode-specific `checkpoint_last` through
`RESUME_CHECKPOINT`.

## Required reporting

For each target variable and UTC hour, report model RMSE and bias, background RMSE and bias,
`1 - MSE(model) / MSE(background)`, and the fraction of validation days on which the model beats
its background. A checkpoint trained with daily 00 UTC backgrounds is structurally loadable in
hourly mode but is not a valid evaluation of the hourly-background experiment.

Hourly files require roughly 9.9 GB per year per raw or normalized copy on the current grid.
Retain raw files until normalization and diagnostics pass; any later deletion should be explicit.
