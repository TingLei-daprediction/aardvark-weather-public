# Blueprint: normalize raw backgrounds in the loader

Status: implementation added locally, September 27, 2026. Cluster baseline approval
and real-checkpoint GPU regression remain pending. See [background_input.md](background_input.md)
for implemented options, checkpoint manifests, and verification commands.

## Decision

Add an optional raw-background mode alongside the existing normalized-file mode.
The existing mode remains the default and fully supported. In raw mode, read
physical-unit background memmaps and normalize each selected frame in the
loader, using the existing target-analysis mean and standard deviation. Preserve
the model input channel order, time selection, and model-facing `climatology_*`
keys. This changes where normalization happens, while preserving its meaning.

Implemented CLI: `--background_input normalized|raw`, default `normalized`.
This is independent of `--background_mode daily_00z|hourly`, which selects cadence.
Thread the option through training entry points and dataset constructors, with
the same backward-compatible default for direct Python callers. Record it in
run configuration/logs and pass it consistently to preflight and diagnostics.

| Cadence | Input option | File stem | Loader action |
| --- | --- | --- | --- |
| daily_00z | normalized (default) | `background_` | Read as-is |
| daily_00z | raw | `background_raw_` | Normalize selected frame |
| hourly | normalized (default) | `background_hourly_` | Read as-is |
| hourly | raw | `background_raw_hourly_` | Normalize selected frame |

Each stem is followed by `rtma_ok_sfc_1_<YYYY>-<MM>.memmap` in the
configured background directory. Never infer the input mode from numerical
ranges or silently fall back to the other file type. Missing selected files
must produce an error identifying the option and expected path.

Do not switch to background-derived statistics in the same change. That would
change the model inputs and require a separate experiment and baseline.

Normalized mode preserves file selection and numerical input values, but now
also writes `background_normalization.json` in the run directory. Both modes
reject mismatched factors in new checkpoints and reject a reused output directory
whose manifest contains different factors. These are intended compatibility
checks, not a promise that every old command has identical side effects or error
behavior. Old checkpoints without metadata still warn in normalized mode.

## Are the statistics scalars or fields?

The current RTMA Oklahoma path uses one scalar mean and one scalar standard
deviation **per variable**, stored as vectors of shape `(5,)`:

| Channel | Variable | Physical units |
| --- | --- | --- |
| 0 | tas, 2 m temperature | K |
| 1 | sh, 2 m specific humidity | kg/kg |
| 2 | psl, surface pressure in this dataset | Pa |
| 3 | u, 10 m zonal wind | m/s |
| 4 | v, 10 m meridional wind | m/s |

These are neither one shared scalar across all variables nor spatial fields.
`scripts/compute_rtma_ok_target_norms.py` reduces over training timestamps and
both spatial axes, with equal weight per gridpoint/time sample. It saves
`norm_factors/mean_rtma_ok_sfc_1.npy` and `std_rtma_ok_sfc_1.npy`.
Its standard deviation is `sqrt(max(E[x²] - E[x]², 0)) + 1e-8`.

For the documented experiment, statistics must use 2022 training data only;
2023 validation data uses those same frozen statistics. Existing cluster files
and their actual values/provenance have not been inspected in this session.

Normalize channel c at each grid point with:

```text
normalized_background[c, x, y] =
    (raw_background[c, x, y] - target_mean[c]) / target_std[c]
```

The divisor is standard deviation, not variance. If variance is the available
quantity, take its square root. Reuse the saved standard deviations here without
adding another epsilon.

The background's own mean/std written by `normalize_background.py` are currently
diagnostic only. Using target statistics keeps background and target in the same
coordinate system: normalized background minus normalized target equals their
physical difference divided by target std. It does not imply the normalized
background itself has exactly zero mean or unit variance.

Spatial normalization would instead need `(5, nlon, nlat)` statistics estimated
over time at each location. It changes the treatment of spatial structure and
is outside this migration. Observation normalization has its own contracts,
including station-dependent monthly statistics; do not change those here.

## Implementation steps

### Normalization source and checkpoint contract

The authoritative raw-loader sources are
`norm_mean_path(self.aux_data_path, self.era5_mode)` and
`norm_std_path(self.aux_data_path, self.era5_mode)` under the active grid-config
templates. Do not substitute `data_path` or hardcoded `norm_factors` filenames.
At startup, log the resolved absolute paths, full per-channel mean/std values,
channel order, arithmetic dtype, input option, and cadence.

The current offline normalizer hardcodes paths under `data_dir`. Add optional
`--aux_data_path` and `--grid_config` support to it, preserving its current paths
when those arguments are omitted. Explicitly configured runs must resolve the
same files as the loader. Parity checks must use that same resolver and verify
the actual factors used to create the comparison files; matching filenames or
shapes alone is insufficient.

Save a versioned normalization manifest in the run directory and in every new
checkpoint used for resume or inference, including best and last checkpoints.
Store resolved paths for provenance, original norm-file hashes, effective
float32 mean/std vectors, channel order, and a hash of those effective vectors
in a canonical dtype/byte order. Cache the validated vectors for the run.
Check this metadata before using a checkpoint for warm start, resume, or
inference. A different path alone is allowed when values agree; an effective
vector/channel mismatch is a hard error. Record file-hash changes even when
the effective float32 vectors are identical. Cadence is recorded separately;
switching raw/normalized representation with verified parity is allowed.

Old checkpoints have no normalization metadata. Preserve existing normalized
workflows with a prominent unverifiable-provenance warning. For raw mode,
require a reviewed external manifest tied to the checkpoint checksum before
using such a checkpoint, including the regression baseline checkpoint. Do not
invent historical provenance by treating whatever files are on disk as verified.
`dev-mul_checkpoint-infer.sh` must run the shared metadata check for each
checkpoint, not only the first. Add mismatch and legacy-checkpoint tests.

Normalized memmaps currently have no embedded provenance. A sidecar containing
the factors and hashes used by `normalize_background.py`, checked by the loader
or preflight, is a useful separate follow-up. Until then, metadata for normalized
mode records configured factors but cannot prove how old memmaps were produced.

### Code and workflow changes

1. **Make raw file paths explicit.** Introduce raw daily/hourly template keys and
   helper names in `aardvark/grid_config.py`, with corresponding entries in
   `grid_config_ok.yaml` and `grid_config_ok_reduced_grid_B.yaml`. Point them at
   existing `background_raw_*` and `background_raw_hourly_*` products. Retain old
   normalized-path helpers for the supported default mode. No automatic
   fallback from raw to normalized files: memmap shape cannot identify units.
2. **Select and normalize in `aardvark/loader.py`.** Select the path helper using
   both options validated in `__init__`, next to the existing `background_mode`
   check, so invalid strings fail immediately. Select paths using
   both cadence and input option. Open backgrounds read-only with the
   existing frame/channel/grid checks. In the monthly branch of `get_index`,
   select the frame, normalize out of place only for `background_input=raw`,
   then convert to the existing tensor
   layout. Daily mode keeps `date.day - 1`; hourly mode keeps `frame_in_month`.
   Normalize only the selected `(C, nlon, nlat)` frame, not a full month.
3. **Validate and cache the factors once.** Require original mean/std shapes
   `(C,)`, finite values, positive std, and channel agreement. Cast loaded
   vectors with `.astype(np.float32)` BEFORE reshaping to `(C, 1, 1)`;
   validate finiteness/positive std again after casting. Use these copies to match
   the current offline normalizer's float32 arithmetic and output. Do not mutate
   target normalization arrays or raw memmaps. `self.means/self.stds` can contain
   difference statistics under `diff=1`; reject `background_input=raw` with
   `diff=1` outright in `__init__`. Preserve existing normalized-mode behavior.
   Leave the non-monthly global climatology path unchanged; reject an explicit
   raw option on unsupported paths rather than silently ignoring it.
4. **Offer both preparation workflows.** Keep `normalize_background.py` and its
   wrappers fully supported. Thread `BACKGROUND_INPUT=normalized|raw` (default
   `normalized`) through build/submission wrappers, translating it to the Python
   option. Normalized mode retains the current normalization jobs and dependencies;
   raw mode skips background normalization and proceeds from raw preparation to
   diagnostics once target norms exist. Apply conditional behavior in
   `run_build_urma_ok_hourly_background.sh`,
   `run_compute_rtma_ok_2022_norms.sbatch`, and `run_prep_urma_ok_hourly.sbatch`.
   A norm update makes existing normalized backgrounds stale; regenerate those
   products before using normalized mode, even if the update ran in raw mode.
   No existing data should be deleted for this addition.
5. **Update consumers with the same option.** `diagnose_hourly_background.py`
   reads physical backgrounds directly in raw mode and retains inverse
   normalization in normalized mode; retain target std for dimensionless RMSE
   thresholds. Audit/update `diagnose_encoder_analysis.py` (including its
   raw-sibling logic) and `check_ok_run_files.py` for both modes with the same
   default and file selection. Update docstrings, grid comments, build messages, and
   `docs/hourly_background_experiment.md`.

### Entry-point audit checklist

At implementation time run `rg -l 'background_mode|BACKGROUND_MODE' .` from the
repository root, including root-level launchers. Trace every caller and subclass
that forwards cadence. Consumers must also forward `background_input`; raw
producers need only document their output contract, and workflow wrappers must
select the appropriate stages. Do not add meaningless flags to producers.

Explicit current audit targets (re-run the search as the repository evolves):

- `aardvark/train_module.py`, `aardvark/loader.py` (base dataset and forwarding
  subclass), and `aardvark/trainer.py` (checkpoint metadata/checks).
- Root `dev-mul_checkpoint-infer.sh`; the `training/new-tl-infer*` launchers.
- `training/new-bsize_2-train-2022-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static-hourly-background.sh`
  and `training/new-bsize_2-train-12month-selected-rtma_ok_sfc-2GPU-lr5e-5-gridB336-static.sh`.
- Both `training/test-*val-regression*.sh` harness scripts.
- `scripts/check_ok_run_files.py`, `scripts/diagnose_hourly_background.py`,
  `scripts/diagnose_encoder_analysis.py`, `scripts/prep_urma_ok_hourly.py`, and
  `scripts/normalize_background.py`.
- `scripts/run_build_urma_ok_hourly_background.sh`,
  `scripts/run_compute_rtma_ok_2022_norms.sbatch`,
  `scripts/run_prep_urma_ok_hourly.sbatch`,
  `scripts/run_prep_urma_ok_hourly_background.sbatch`,
  `scripts/run_normalize_urma_ok_hourly_background.sbatch`, and
  `scripts/run_diagnose_urma_ok_hourly_background.sbatch`.

## Verification and rollout

1. **Merge gate: an approved baseline must exist before any loader change is
   merged.** None exists yet. First complete `docs/regression_test_handoff.md`:
   parent-shell conda activation, `--seed 42`, and stale-comment corrections are
   committed as A (`23b19c8`). Black-only formatting is isolated in `958e56f`;
   implementation and tests are in the following feature commit B. Run B's unit
   tests in the pinned cluster `aardvark-env` before any regression runs; this is
   still pending. Use separate detached A/B worktrees with explicit checkout and
   commit guards, as documented in the handoff. Capture and approve the baseline
   from A with the old loader and frozen statistics. Keep checkpoint, norms, raw data,
   GPU/software configuration, seed, and training settings fixed.
2. Add a small loader-level test using distinct known values per channel and
   spatial position. Assert expected normalized values, float32 dtype, shape,
   and unchanged source bytes after repeated reads. Include mean-valued inputs
   mapping to zero and mean-plus-std inputs mapping to one. For exact assertions,
   use exactly representable values such as mean=2.0 and std=0.5; arbitrary
   float32 mean-plus-std expressions can round and need an appropriate ULP check.
3. Exercise all four cadence/input combinations, including a month boundary.
   Verify omission of the new option reproduces existing behavior and that
   normalized mode never normalizes a second time. Confirm the
   actual `climatology_*` tensor delivered downstream matches the expected frame
   and channel order. Check missing raw files and invalid statistic shapes/values
   fail clearly rather than silently accepting legacy normalized products.
4. On representative real frames, compare new loader outputs to existing offline
   normalized outputs generated from the **same raw files and same norm files**.
   Resolve those norm files through the loader's aux-data/grid-template helpers,
   including tests with distinct data/aux roots and overridden YAML norm paths.
   Float32 operation parity should allow exact equality; investigate differences
   before choosing any nonzero tolerance. Include both background cadences.
5. Run a fixed-checkpoint forward comparison and the one-GPU regression harness.
   The existing harness defaults to `daily_00z`; it does not cover hourly training.
   Add input-option forwarding with the default unchanged, then run the candidate
   code twice in fresh directories: once with default normalized input and once
   with raw input. Compare BOTH against the SAME approved daily baseline, using
   its seed, checkpoint, factors, and calibrated tolerances. These two runs do
   not establish hourly end-to-end coverage: this initial rollout relies on
   loader-level parity and fixed-checkpoint forward comparisons for hourly mode.
   A later hourly training regression requires its own calibrated baseline.
   Check hourly diagnostics in physical
   units; small rounding changes from removing inverse normalization are possible.
6. Document opt-in raw commands alongside the unchanged normalized commands.
   Preserve old configurations and file paths. Keep the
   existing approved baseline if parity holds; do not automatically regenerate it.

Acceptance: both input options work for both cadences; default file selection and
numerical inputs are preserved, with the intended manifest writes and mismatch
errors described above. Raw mode requires no normalized background files in loading,
preflight, or diagnostics and skips their generation. Model inputs retain the
existing numerical meaning across modes. Raw-mode checkpoint loads enforce the
normalization manifest contract, including legacy checkpoint handling. Startup
logs identify the actual factors; mismatch tests pass. The approved daily baseline
passes in both modes; hourly coverage is reported with the limits above.
