# Hourly analysis increment target implementation plan

Approved 2026-10-12 after review. This record incorporates the final restriction
that validation divisibility applies only to monthly RTMA assimilation.

## 1 Configuration and scope

Add `--target_mode analysis|increment` (default analysis). Increment is
`analysis(H) - raw_background(H)`. Require monthly RTMA surface assimilation,
hourly backgrounds, 1H cadence and diff=0 in the dataset constructor.
Finetuning, downscaling and E2E reject increment mode. Architecture is unchanged.

## 2 Preprocessing integrity

Generate separate float32 monthly increments from raw hourly backgrounds, either
during GRIB preparation or from existing memmaps. Add a distinct grid-config
template. Check matching shape, channel order, SW origin and axis order through
upstream sidecars. Default expected channels are tas, sh, psl, u, v. Missing legacy
sidecars require an explicit override, recorded as channel_order_source=override.
Propagate NaNs: the increment mask equals the union of source masks; reject
infinities. Write temporary files, reopen, verify sampled first/last/intermediate
frames bit-exactly against float32 subtraction (NaN masks separately), then
publish atomically. Monthly sidecars record sign, source paths/hashes, channels,
grid orientation, shape, frame count, time layout, provenance and tool version.

## 3 Statistics and startup

Compute statistics from increment files, all frames in selected training months,
NaN-aware, float64 accumulation. Save separate increment mean/std plus metadata
with source hashes, months, channels and valid counts. Require 99% valid coverage
relative to analysis-valid points and std_increment >= 1e-3 * std_analysis.
Reject invalid factors rather than clamp; check float32 representability. Log
analysis/increment NaN fractions. Manifests are optional for training: derive
effective training months from the dataset index when absent. Training months
must be covered by statistics months. Overlap warns as in analysis mode and
continues. Startup checks sidecars, sizes and recorded hashes; full rehashing is
opt-in. Validation months need not be covered by the statistics inputs.

## 4 Loader

Select files/factors by target mode. Reuse self.means/self.stds for targets;
self.background_means/self.background_stds always use analysis factors.
Normalize (increment - increment_mean) / increment_std. Guard
unnorm_base_context against increment factors. Reconstruct using the exact
target hour's background, supporting raw and normalized background inputs.

## 5 Metrics outputs and checkpoints

Physical analysis prediction is denormalized output in analysis mode and
background + denormalized increment in increment mode. Per-channel RMSE sums
squared errors and valid counts over the entire validation set, all-reduces
across DDP ranks, then takes one square root, uniformly weighted on the grid.
Both modes use identical samples and the intersection of valid analysis and
background points, with coverage reported. Non-finite predictions inside that
fixed mask fail evaluation after cross-rank aggregation. Background masks
affect analysis-mode metrics only, never its training loss. Cross-mode comparisons
use physical RMSE; normalized loss can still select checkpoints within a run.
Only monthly RTMA assimilation requires validation size divisible by world_size
to avoid DistributedSampler padding duplicates. Other modes are unaffected.
Save explicitly named physical increments and reconstructed analyses, including
inference/epoch 0. Increment starts from scratch: every subsequent checkpoint
load (resume, inference, epoch 0, initialization) requires matching increment
target metadata, factor hashes and effective hash. No analysis initialization
override. Metadata includes factor paths, channels, statistics months and override
provenance. Document altered effective channel weighting and log physical RMSE.

## 6 Integration and validation

Extend hourly build/preflight wrappers and training/inference banners/options.
Use synthetic tests for sign, month ends (31 days, Feb 28/29), channel metadata,
round trips, zero normalized output, background modes, masks, month coverage,
checkpoint rejection, constructor gates and distributed metrics. Actual archive
processing/training requires the external HPC data environment.

TODO (deferred, out of scope): A future pipeline-wide consistency tool will check
unit and scale consistency across all data products, including observations, by
comparing statistics. Example diagnostics: increment std relative to analysis std
and increment mean relative to analysis std. Unit consistency between analysis
and background is the preprocessing pipeline's responsibility. This implementation
assumes consistent units and adds no units fields, unit checks or unit-mismatch tests.

## Acceptance criteria

- Default analysis-target behavior remains compatible.
- Constructor rejects increment with diff=1, daily_00z, 15min or non-RTMA obs.
- Finetune, downscaling and E2E explicitly reject increment; no architecture change.
- Increment paths do not collide with analysis/background paths.
- Raw hourly inputs match shape, channel order, orientation and frame layout.
- Channel mismatch fails; missing sidecars require recorded override provenance.
- Override provenance is carried into checkpoint target metadata.
- Reopened first/last/intermediate frames match subtraction bit-exactly.
- Increment NaN mask equals source-mask union; infinities fail.
- Interrupted writes do not publish partial products.
- Statistics use actual increments, every training frame and float64 accumulation.
- Coverage >=99% is measured against analysis-valid points; NaN fractions logged.
- Analysis-relative std floor and float32 factor validity fail without clamping.
- Effective training months are covered by statistics months, with or without manifests.
- Train/validation overlap warns and continues as in analysis mode.
- Startup checks sidecars, sizes and recorded hashes; rehashing is opt-in.
- Target factors reuse means/stds; background always uses analysis factors.
- unnorm_base_context never applies increment factors to background.
- Both modes report full-validation per-channel physical RMSE and coverage.
- Uniform metrics all-reduce squared sums/counts and take one square root.
- Cross-mode comparisons use identical samples and a fixed source-valid intersection.
- Non-finite predictions inside that intersection fail after cross-rank aggregation.
- Divisibility check applies only to monthly RTMA assimilation; other modes unaffected.
- Non-divisible monthly RTMA validation fails clearly using actual world_size.
- One/two-rank physical RMSE agrees within floating tolerance on identical samples.
- Increment rejects missing/mismatched checkpoint metadata for every loading purpose.
- Outputs distinguish physical increments and reconstructed analyses.
- bg=analysis-1 produces +1 and reconstructs analysis.
- Boundary tests include 31-day months and Feb 28/29.
- Permuted channel metadata fails.
- Increment normalization round trip is accurate within float32 tolerance.
- Zero normalized output reconstructs background + increment_mean.
- Raw/normalized backgrounds reconstruct equivalently within float32 tolerance.
- Tests cover invalid configuration, coverage, month and checkpoint cases.
- Documentation explains channel weighting and records the deferred consistency TODO.

## Implementation review amendments

- Current analysis and raw-input background sidecar hashes must match the increment's
  recorded sources at startup; explicitly overridden legacy sources without sidecars warn.
- Physical metric fields and analysis witness reads are validation-only.
- Coverage >=99% and std >=1e-3 * analysis std are hard generation/load limits.
- Constructor-level train/validation metadata and existing-run JSON collision checks
  are exercised in tests, with only GPU placement/DDP wrapping replaced for CPU execution.
