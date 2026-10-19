# Hourly increment targets

The approved design and acceptance criteria are recorded in
[the implementation plan](hourly_increment_target_plan.md).

Use `--target_mode increment --background_mode hourly --time_freq 1H --diff 0`
with `--mode assimilation --obs_set rtma_surface`. Analysis remains the default.
No architecture changes are required. Increment experiments start from scratch;
analysis-mode weights cannot initialize them. Matching increment checkpoints can
subsequently be used for resume and inference.

## Prepare the files

Increment files contain physical float32 `analysis(H) - raw_background(H)`, with
shape `(days*24, 5, nlon, nlat)`, in channel order `tas, sh, psl, u, v`.
The regional YAMLs use `urma/increment_rtma_ok_sfc_1_1h_YYYY-MM.memmap`.
Files have `.memmap.json` sidecars with SW origin, axis directions/order, UTC
month/frame layout, sign convention, source hashes and channel provenance.

To generate increments from existing raw monthly products:

```bash
python scripts/build_hourly_increments.py \
  --data_root /path/to/data \
  --grid_config aardvark/grid_config_ok_reduced_grid_B.yaml \
  --months_file training/rtma_ok_train_months_2022.txt

python scripts/build_hourly_increments.py \
  --data_root /path/to/data \
  --grid_config aardvark/grid_config_ok_reduced_grid_B.yaml \
  --months_file training/rtma_ok_val_months_2023.txt

python scripts/compute_increment_norms.py \
  --data_root /path/to/data \
  --grid_config aardvark/grid_config_ok_reduced_grid_B.yaml \
  --months_file training/rtma_ok_train_months_2022.txt
```

Existing products without source sidecars require `--allow_legacy_inputs` on the
build command. This explicitly assumes the expected channel order, SW grid
orientation and hourly layout; it does not infer them from file size. The
override is recorded in increment sidecars, statistics and checkpoint metadata.
Source sidecars that exist but disagree are always rejected. Regeneration
requires `--overwrite`.

For new GRIB preparation, add `--write_increments --background_mode hourly` to
`scripts/prep_urma_ok_hourly.py`. It writes source sidecars and verified increments
after each completed month. With `--background_only`, analyses must already exist;
missing legacy analysis sidecars still require the explicit override.

For the existing HPC job chain:

```bash
TARGET_MODE=increment bash scripts/run_build_urma_ok_hourly_background.sh
```

The final dependent job builds train/validation increments and training-only
statistics. Optional environment variables are `INCREMENT_TRAIN_MONTHS_FILE`,
`INCREMENT_VAL_MONTHS_FILE`, `GRID_CONFIG`, `DATA_ROOT`, and `AUX_DATA_ROOT` for
that increment job.
For legacy analysis products, explicitly set `ALLOW_LEGACY_INCREMENT_INPUTS=1`.
Review the quality reports and wait for the increment job to succeed before
training. Background normalization remains independent of increment generation.

## Statistics and loading

The statistics tool reads every frame of the selected increment months, using
float64 centered moment accumulation. It writes:

- `norm_factors/mean_increment_rtma_ok_sfc_1.npy`
- `norm_factors/std_increment_rtma_ok_sfc_1.npy`
- `norm_factors/mean_increment_rtma_ok_sfc_1.npy.json`

The sidecar records months, increment hashes, channel order, valid counts and
NaN fractions. Valid increment coverage must reach 99% of analysis-valid points
per channel. Permanently masked analysis points do not reduce this coverage.
The increment std must be at least `1e-3 * analysis_std`; invalid factors fail
rather than being clamped. Preprocessing options can tighten these limits, but
cannot lower them. Loading factors independently rechecks both hard limits.
Use `--aux_data_path /path/to/aux` on the statistics command when factors are
stored separately from increment data; it defaults to `--data_root`.

Targets use `(increment - increment_mean) / increment_std`. The loader reuses
`self.means/self.stds` for the selected target. Backgrounds always use analysis
factors, including with `--background_input raw`. Increment scaling changes each
channel's effective loss weight, so compare runs using physical RMSE.

Training-month manifests are optional. Effective training months must be covered
by statistics months; validation months need not be. Train/validation or
statistics/validation overlap warns and continues. Eval-only timestamps (`--epoch
0`) are not checked as fitting months. Startup checks increment sidecars, sizes
and recorded training-source hashes. Current analysis sidecar hashes must also
match the sources recorded when increments were built; raw background inputs
receive the same check. Missing source sidecars warn only for explicitly
overridden legacy sources. `--verify_increment_hashes` additionally rehashes
increment contents and the checked source files.

Physical truth/background tensors are enabled only for validation through
`include_physical_targets=True`. Training does not read the extra analysis witness
or copy physical metric fields to the device. Validation caches analysis memmaps
instead of reopening them for each sample.

## Launch and evaluate

```bash
TARGET_MODE=increment sbatch --export=ALL \
  training/new-bsize_2-train-2022-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static-hourly-background.sh

TARGET_MODE=increment BACKGROUND_MODE=hourly sbatch --export=ALL \
  training/new-tl-infer-encoder-rtma_ok_sfc-gridB336.sh
```

Edit the inference script's checkpoint and timestamp first. Increment run
directories have an `-increment` suffix. Resume uses `RESUME_CHECKPOINT` and
requires compatible target metadata, effective factors and factor-file hashes.
Finetuning, downscaling and E2E reject increment targets/checkpoints.

The physical prediction is `background + output * increment_std + increment_mean`.
Zero normalized output therefore predicts `background + increment_mean`.
Both target modes report uniformly weighted per-channel physical analysis RMSE:
sum squared errors and valid counts over the entire validation set, all-reduce
across ranks, then take one square root. Analysis/background validity defines a
fixed metric mask; background validity never changes analysis-mode loss. NaN or
Inf predictions inside this mask are counted across ranks and fail evaluation
after aggregation, rather than disappearing from the score. Non-finite predictions
outside this mask do not affect the score. Cross-mode comparisons use identical
validation samples and source-valid points, with coverage/counts reported.

Monthly RTMA validation sample counts must divide evenly by `world_size`, to
avoid duplicated samples from `DistributedSampler` padding. This restriction
does not affect forecast/processor or other training paths. Single-time inference
uses one rank.

Metrics are saved in `physical_analysis_metrics.json` and per-epoch JSON files.
In increment mode, last-batch physical outputs are:

- `unnorm_increment_preds_RANK.npy` and `unnorm_increment_targets_RANK.npy`
- `unnorm_analysis_preds_RANK.npy` and `unnorm_analysis_targets_RANK.npy`

Analysis mode retains `unnorm_preds_RANK.npy` / `unnorm_targets_RANK.npy`.
Monthly RTMA `rmse_RANK.npy` now contains pooled, uniformly weighted physical RMSE;
it is not directly comparable to RMSE arrays from earlier implementations.
Physical outputs contain only the requested `var_start:var_end` channels.
These outputs cover the last validation batch, following the existing inference
convention; aggregate metrics cover the full validation set. Normalized increment
debug outputs have the `normalized_increment_` prefix.
For `diagnose_encoder_analysis.py`, pass the reconstructed analysis filenames
through `--pred_file` and `--target_file`, plus `--background_mode hourly`.

## Validation

Run `python scripts/check_increment_targets.py` in the PyTorch environment.
This exercises real loader batches, preprocessing CLIs, loss/normalization,
checkpoint gates and one/two-rank Gloo trainer evaluation on synthetic months.
Full GRIB/archive processing and GPU training require the external HPC environment.
The deferred pipeline-wide unit/scale consistency TODO is recorded in the plan;
this feature assumes source units are consistent.

Initial implementation validation on 2026-10-18: all nine synthetic regression groups
passed using Python 3.12 and CPU PyTorch 2.5.1, including one/two-rank trainer
evaluation in both target modes. Changed Python files passed compilation and
the six affected launch/build scripts passed Bash syntax checks. Full GRIB
decoding, the original Python 3.8 environment and CUDA/NCCL training were not run.

The subsequent static review added source-sidecar consistency checks,
validation-only metric fields, explicit failure on non-finite predictions, hard
normalization limits at load time, a separate auxiliary root for statistics,
and regression coverage of the trainer constructor's metadata checks. Preflight
date errors now fail cleanly, publication messages follow file publication,
and analysis-mode plotting hints are retained.

Follow-up validation passed all 17 regression groups, including successful full
preflight checks for synthetic analysis and increment training scripts, malformed
date rejection, and a clear error when physical validation fields are disabled.
The date guard resides in preflight `main()`, not in the reusable `month_range()`
helper. Running preflight against the actual hourly HPC training wrapper parsed
all 2022/2023 months and then stopped cleanly at the unavailable external grid
files. Black checks passed for all 14 feature Python files; compilation and
whitespace checks also passed.
