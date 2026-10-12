# September 2026 Progress Report: Aardvark-Based RTMA

**Reporting period:** September 1–28, 2026  
**Scope:** Oklahoma regional surface analysis using Aardvark, surface observations,
and URMA analysis/first-guess data.

## Summary

September's work strengthened the data pipeline, experimental controls, and
verification framework for Aardvark-based RTMA. Building on the existing
five-variable surface-analysis configuration, development added hour-matched
background support, more reproducible training and checkpoint evaluation, and an
optional path for normalizing raw backgrounds during data loading. A regression
framework and nine passing local CPU tests now support further development.
Validation in the pinned cluster environment and quantitative analysis-skill
comparisons remain the next milestones.

## Main accomplishments

### 1. Hourly background support and experiment design

- Added support for hourly URMA first-guess backgrounds alongside the existing
  daily 00 UTC background. Hourly mode aligns backgrounds, observations, and
  targets using the same timestamp indexing.
- Added preparation and Slurm workflow scripts for the 2022–2023 hourly archive,
  with separate filenames for daily and hourly products.
- Defined a controlled daily-versus-hourly experiment: train on 2022, validate
  on 2023, and estimate normalization factors from training data only. Both
  configurations use matching training settings and a fixed seed.
- Added diagnostics that report background bias and RMSE by variable and UTC
  hour, including checks for suspicious agreement with current or previous
  analyses that could indicate data leakage or timing errors.

These changes provide the tools to evaluate whether hour-matched background
information improves the regional analysis. Experiment results are still pending.

### 2. Training continuity and checkpoint evaluation

- Added explicit training/validation month manifests and validation subsampling
  that cycles through UTC hours.
- Improved restart support through a saved latest checkpoint containing model,
  optimizer, scheduler where applicable, epoch, and best-loss state.
- Updated the multi-checkpoint inference launcher to apply consistent inference
  settings, isolate outputs, and record checkpoint-to-output mappings.

This makes interrupted experiments easier to resume and checkpoint comparisons
easier to trace.

### 3. Optional normalization of raw backgrounds in the loader

- Implemented `--background_input raw|normalized`, with `normalized` retained as
  the default. Both daily and hourly backgrounds support either input format.
- Raw mode reads separately named physical-unit files and normalizes each selected
  frame with the existing target-analysis mean and standard deviation. There is
  one scalar mean/std pair per variable, shared across the spatial grid.
- Updated preparation workflows to skip offline background normalization in raw
  mode, while retaining the existing normalized-file workflow.
- Added normalization metadata to run outputs and new checkpoints, with checks
  that reject incompatible factors during checkpoint loading. Legacy checkpoints
  used in raw mode require a reviewed, checkpoint-specific manifest.
- Incorporated colleague review feedback on norm-file resolution, float32
  arithmetic, checkpoint provenance, inference manifests, and preflight checks.

Raw mode removes the need to generate a second, normalized copy of background
files. Normalized mode preserves its input values and file selection, while
gaining intentional metadata writes and compatibility checks.

### 4. Regression testing and review readiness

- Added compact JSON summaries and comparison tools for validation loss, training
  loss, per-variable RMSE, and selected prediction/target checks.
- Prepared a single-GPU baseline procedure with a fixed checkpoint and seed,
  repeated unchanged-code runs, and tolerance calibration.
- Added checkout-directory and commit checks to prevent accidental baseline
  capture with the candidate loader.
- Separated the work into a harness-only baseline commit, a formatting-only
  commit, and a feature commit, making the functional changes easier to review.
- Passed nine local CPU tests covering raw/normalized equivalence, daily/hourly
  indexing across a month boundary, norm-path overrides, source-file preservation,
  invalid settings, checkpoint compatibility, and diagnostic consistency.

### 5. Repository organization and collaboration

Created a pull request to AI-RTMA adding `ai-rtma-bundle`, with the RTMA-adapted
Aardvark repository included as a Git submodule. The PR demonstrates the proposed
AI-RTMA repository structure so colleagues can review how the components should
be organized and how Aardvark fits into the broader project.

## Validation status and remaining work


| Item | Status as of September 28 |
| --- | --- |
| Hourly background and controlled-experiment workflows | Implemented |
| Optional raw-background loading and provenance checks | Implemented and committed locally |
| Local CPU fixture tests | Nine passed |
| Python and shell syntax checks | Passed |
| Unit tests in the pinned cluster `aardvark-env` | Pending |
| Approved regression baseline from the old loader | Pending |
| Normalized/raw candidate GPU comparisons | Pending |
| Daily-versus-hourly analysis-skill evaluation | Pending |

Local tests used a newer Python/NumPy/PyTorch environment than the project's
pinned cluster environment. The pinned-environment tests and real-data GPU
comparisons are required before merging the loader change. This report does not
establish an improvement in analysis accuracy or operational readiness.

## Priorities for October

1. Run the unit suite in the cluster's pinned environment and resolve any
   version-specific issues.
2. Capture and approve the old-loader baseline from commit A (`23b19c8`), then
   compare normalized and raw candidates from commit B (`740b59e`) against it.
3. Complete the controlled daily-versus-hourly background experiment after the
   data-quality checks pass.
4. Report model and background RMSE/bias by variable and UTC hour, skill relative
   to the background, and the fraction of validation days improved by the model.

## Supporting records

This report is based on September's repository history and the project records
available through September 28. Cluster execution logs and quantitative training
results were not available for verification in this reporting session.

- [Hourly-background experiment](hourly_background_experiment.md)
- [Raw and normalized background usage](background_input.md)
- [Regression baseline and candidate procedure](regression_test_handoff.md)
- Key commits: `23f51f1` (hourly support), `85bd611` (checkpoint inference),
  `40b782c` (compact regression summaries), `23b19c8` (baseline harness),
  `958e56f` (formatting), and `740b59e` (raw-background feature).
