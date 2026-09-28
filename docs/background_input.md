# Raw and normalized background input

The default remains `--background_input normalized`: the loader reads existing
normalized background files unchanged. Select `--background_input raw` to read
physical-unit files and normalize each frame in the loader. Both choices work
with `--background_mode daily_00z` and `--background_mode hourly`.

| Cadence | Normalized file stem | Raw file stem |
| --- | --- | --- |
| daily_00z | `background_` | `background_raw_` |
| hourly | `background_hourly_` | `background_raw_hourly_` |

Each stem precedes `rtma_ok_sfc_1_YYYY-MM.memmap`. The Oklahoma YAMLs place them
under `urma/`; built-in defaults use `era5/`. The new `background_raw_month` and
`background_raw_hourly_month` templates can be overridden independently. There
is no fallback between representations.

## Run options

Direct training/inference via `aardvark/train_module.py`:

```text
--background_input raw --background_mode hourly
```

Raw input is supported for monthly RTMA surface assimilation only. It rejects
`diff=1`. Daily backgrounds also support the existing 15-minute RTMA path;
hourly backgrounds require `time_freq=1H`.

The updated RTMA A/B training launchers, `training/new-tl-infer*` launchers,
root `dev-mul_checkpoint-infer.sh`, and regression launcher accept:

```bash
export BACKGROUND_INPUT=raw
# Submit the selected launcher with sbatch as usual.
```

The two A/B training launchers add `-raw` to the output directory in raw mode.
Other launchers retain their configured output paths; choose a fresh directory
for comparisons. New checkpoints store normalization metadata automatically.

## Statistics and checkpoint compatibility

Background normalization is `(raw - target_mean) / target_std`, using one scalar
mean/std per channel, broadcast over the grid. Factors are resolved from
`--aux_data_path` through the active grid configuration's `norm_mean` and
`norm_std` templates. Raw arithmetic uses float32, matching the offline script.
Startup logs include resolved paths, values, channel order, and hashes.

The trainer saves `background_normalization.json` and includes the same metadata
in best/last checkpoints. Warm start, resume, and inference fail before loading
weights if effective factors or channel order differ. Moving unchanged factors
to another path is allowed. The run caches its factors in memory.

These checks also apply to normalized mode. Its default file selection and
numerical inputs are preserved, but it now writes the manifest and rejects both
checkpoint mismatches and reused output directories with different manifest
factors. Those additional writes and errors are intentional changes.

Old checkpoints have no such metadata. Normalized mode retains compatibility
with a warning. Raw mode requires a reviewed manifest tied to the checkpoint's
SHA256. After verifying the **original run's** factors from retained data/logs,
record that evidence:

```bash
python scripts/record_background_norm_manifest.py \
  --checkpoint /path/to/epoch_123 \
  --aux_data_path /path/to/verified-original-aux-data \
  --grid_config aardvark/grid_config_ok.yaml \
  --reviewed_source "Describe the archived run and evidence identifying its factors"
```

This writes `/path/to/epoch_123.normalization.json` without overwriting. It does
not recover or prove historical statistics. Pass it with
`--background_norm_manifest /path/to/epoch_123.normalization.json`, or set
`BACKGROUND_NORM_MANIFEST` for a launcher. With a single checkpoint, the inference
launcher accepts that override or discovers its `.normalization.json` sibling.
With multiple checkpoints, it ignores the shared override with a notice and uses
each checkpoint's own sibling when present. Embedded checkpoint metadata needs
no sibling file. Every checkpoint is validated separately.

Normalized files have no provenance sidecars yet: metadata records configured
factors, but cannot prove how old normalized memmaps were made. If factors change,
regenerate those files before using normalized mode. Always keep training-only
factors fixed across validation, resume, and inference.

## Preparation and diagnostics

For raw input, the hourly build chain skips offline normalization:

```bash
BACKGROUND_INPUT=raw bash scripts/run_build_urma_ok_hourly_background.sh
```

Omitting that variable keeps the existing normalized workflow. Both still require
target statistics. Raw preparation scripts always write physical-unit files;
their output contract is independent of the loader choice.

The offline normalizer now supports `--aux_data_path` and `--grid_config` so it
can use the same factors as the loader. Without these flags it retains its
original norm paths. Its data-file naming options remain `--subdir`, `--raw_name`,
and `--out_name`; match those to any custom background YAML templates.

Both Python diagnostics accept `--background_input` and `--aux_data_path`.
`diagnose_encoder_analysis.py` also accepts `--background_mode`. Raw diagnostics
use physical values directly; normalized diagnostics retain inverse normalization.
`check_ok_run_files.py` reads the input option from the launcher, including
`BACKGROUND_INPUT` environment expansion.

## Verification and cluster gate

Run the CPU fixture suite in the project environment:

```bash
python -m unittest discover -s tests -v
```

It checks offline/loader parity for both cadences across a month boundary,
separate data/aux roots and YAML overrides, repeated reads without mutation,
invalid options/factors, checkpoint metadata and legacy manifests, and diagnostics.
The small convolution consumer tests the tensor interface; it is not an actual
trained Aardvark forecast comparison.

The real cluster baseline is still required before merging the loader change.
See [regression_test_handoff.md](regression_test_handoff.md) for exact worktree
and submission commands. First run this suite in the cluster's `aardvark-env`
(Python 3.8 / NumPy 1.24.3 / PyTorch 2.3.0); this check is still pending.
Capture the baseline from commit A (`23b19c8`, harness-only changes and old loader).
Formatting is isolated in `958e56f`; the following feature commit is B. The updated
regression launcher omits new flags in default mode to support A. Its checkout
directory and expected commit can be pinned to prevent accidental execution of B.
After approval, run the new code in separate normalized and raw output directories
against the same daily baseline. Raw warm start requires the reviewed legacy
manifest if that fixed checkpoint predates metadata. Hourly training needs its
own future calibrated baseline; CPU parity does not establish GPU determinism.

Local verification on September 27, 2026: all nine CPU tests passed on Windows
with Python 3.12, PyTorch 2.14 CPU, NumPy 2.5.3, and pandas 2.2.3 in temporary
test dependencies. Changed Python files were formatted with Black 24.4.2 and
parsed for Python 3.8 syntax; 13 changed shell scripts passed a Bash grammar
syntax check. The training CLI help exposes both new options. These checks do
not replace validation in the pinned cluster environment.
