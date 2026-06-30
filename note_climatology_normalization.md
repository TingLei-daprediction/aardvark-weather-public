# Climatology normalization: finding + why the RTMA 00z background will be normalized

## The issue

In the current Aardvark pipeline the **climatology channel is fed to the encoder RAW
(unnormalized), while the target and observations are NORMALIZED.** This is an inconsistency
(very likely an oversight), confirmed in code:

| Field | Built / stored as | Normalized at runtime? | Net scale into the model |
|---|---|---|---|
| Target (ERA5) | raw memmap | **yes** -- `norm_era5` = `(x-mean)/std` (`loader.py` `load_era5_time`) | normalized, O(1) |
| Observations | raw memmap | **yes** -- `norm_data` per modality | normalized, O(1) |
| Climatology | raw daily mean | **no** | **raw**, O(100..1e5) |

Evidence:
- `scripts/build_climatology.py` averages the **raw** ERA5 memmaps and writes the mean directly
  -- there is no `(x-mean)/std` step. Output is raw physical units.
- `loader.py` opens `climatology_data.mmap` and feeds it via `to_tensor(...)` with **no**
  normalization; it never applies `self.means/self.stds` to it.
- Climatology is **input-only** (concatenated into the encoder `encodings`); it is NOT added
  back to the output as a baseline, so there is no raw-space residual trick hiding the mismatch.

## Why it still "works" in current runs

A neural network does **not** require normalized inputs to produce a normalized output -- input
normalization is an *optimization convenience*, not a correctness requirement.

- The first learnable layer (ViT `PatchEmbed`, a Conv2d init ~0.02) can **absorb the scale**: it
  learns a tiny weight on the raw climatology channel (e.g. ~0.003) so a raw ~290 contributes
  like a normalized ~1.
- The `LayerNorm` inside every transformer block re-normalizes activation magnitudes downstream.

So the model trains *despite* the raw channel -- because of network robustness, not because the
raw input is correct.

## Why it is still a real problem

- **At initialization the raw channel dominates.** With patch-embed weights ~0.02, a raw
  climatology value (~290, or ~1e5 for pressure) overwhelms the O(1) normalized channels by
  orders of magnitude. Early training is skewed toward climatology; gradient steps are spent
  learning to down-weight it.
- **Bad interaction with weight decay + a shared learning rate.** The very small weight the
  network needs on the raw channel is penalized/updated on the same scale as all other weights
  (the standard reason input scale matters; cf. `plan_static.md`).

Net: works, but slower/less stable to train and potentially worse-converged than it should be.

## Decision: use a NORMALIZED field for the RTMA 00z background

For the RTMA OK 15-min per-month **00z background** (the climatology slot), we will feed a
**normalized** field, normalized with the **target's per-channel mean/std** (the background has
the same 5 surface fields as the target). Reasons:

1. **Consistency** -- the background then sits on the same O(1) scale as the normalized obs and
   target, so the encoder sees a balanced input set.
2. **Training stability/speed** -- avoids the init-domination and weight-decay/LR issues above.
3. **Correct comparability** -- a normalized background is directly comparable to the normalized
   target the model predicts, which is the natural anomaly/teacher signal.
4. **Don't propagate the oversight** -- the new path should not replicate the raw-climatology
   inconsistency.

### How (preferred): normalize at build time

Store the background already normalized in the per-month file:
`background = (raw_background - target_mean) / target_std`, using the **target** mean/std for the
same fields. The loader stays unchanged (it feeds the climatology channel as-is), and the file is
self-describing. File: `era5/background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap`, shape
`(days_in_month, 5, nlon, nlat)`.

### Alternative: normalize in the loader

Add `climatology = self.norm_era5(climatology)` on the monthly path. One line, keeps the file
raw, but is a behavior change for that channel.

## Note on existing (global) runs

The global runs still feed a **raw** climatology. This is a latent suboptimality, not a crash.
Optionally we could normalize the climatology in the loader for all runs to fix it everywhere --
but that changes existing-run behavior, so it is a separate, deliberate decision. The RTMA
background will be normalized regardless.

## Verify your actual file
`climatology_data.mmap` per-channel stats: physical magnitudes (~288 K, ~1e5 Pa) => raw;
mean ~0 / std ~1 => already normalized.
