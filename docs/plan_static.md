# Static-Parameter Design (Encoder) — Decision Summary

Context: porting Aardvark toward an AI RTMA/URSA system on a 2.5 km Oklahoma domain.
This note covers the **static (elevation) input fields** for the **encoder** step, and
records corrections to the earlier "4 → 6 static channels" plan.

## Decisions

### 1. Do NOT add terrain-gradient channels (`dz/dx`, `dz/dy`)

The static set stays at **4 channels**: orography, land-sea mask, sin(lat), cos(lat).

Rationale: the encoder is convolutional, and a convolution already *is* a learned
spatial-derivative stencil. Precomputed gradients carry no information beyond the
orography field itself, while introducing three hazards that don't exist for scalar fields:

- **Resolution dependence** — elev is resampled to the internal grid in the model
  (`models.py:334`, nearest-neighbor by default); a gradient computed on the native grid
  does not survive resampling (`gradient(resample(z)) != resample(gradient(z))`).
- **Units** — `np.gradient` in index space gives "m per cell," anisotropic on lat/lon grids.
- **Sign/axis convention** — gradients are direction-sensitive and must match the stored
  lat ordering; orography (a scalar) was immune to this.

=> Let the model learn slope features from orography directly.

### 2. Normalize orography with a FIXED divisor (cross-domain consistent)

Instead of per-domain z-score, divide orography by a fixed physical constant
(e.g. `orog / 3000`, or express in km via `orog / 1000`).

Rationale:

- Per-domain mean/std maps the *same elevation* to *different* normalized values on
  different domains — breaks cross-domain meaning.
- A fixed divisor keeps a given elevation mapped to the same value on any domain, is O(1)
  for typical terrain, and degrades gracefully if a new domain has taller peaks (no clipping).
- Mean-centering is unnecessary (a learned bias absorbs any constant offset); only **scale**
  matters, because input scale interacts badly with weight init, a shared learning rate,
  and weight decay.
- Land-sea mask (0–1) and sin/cos lat ([-1,1]) are already bounded and physically defined —
  leave them untouched.

## Corrections to the previous 8-step plan

| Item | Previous plan | Correction |
|---|---|---|
| 1. `build_elev_vars.py` -> 6 channels | Add dz/dx, dz/dy, output `(6, lat, lon)` | **Keep `(4, lat, lon)`.** Still needs: OK-domain orography/LSM input, OK grid files, **res-aware output filename** (currently hardcodes `elev_vars_1.npy` at `build_elev_vars.py:85`), and the **fixed-divisor normalization** of orography. |
| 2. `run_build_static.sh` | Point at OK inputs + new gradient vars | Still update OK paths/grid/output dir; **no gradient variable names**. |
| 3. `--in_channels += 2` everywhere | Increase by 2 | **Dropped** — static count unchanged. |
| 4. Processor `39 -> 41` | "30 ERA5 + 4 static + 5 time = 39 -> 41" | **Dropped, and the arithmetic was wrong.** `ForecastLoader` uses **4** time channels (`loader.py:1936-1952`), so the strict composition is `30 + 4 + 4 = 38` (confirmed by `expected_in_channels_forecast`, `train_module.py:101-103`) — not 39. The published `train_processor_4u_sfc.sh` value of `41` is already inconsistent with the loader and should be reconciled to **38** regardless of this change. |
| 5. Encoder `--in_channels += 2` | Increase by 2 | **Dropped** — unchanged. |
| 6. Decoder/forecast/e2e `+2` | Increase by 2 | **Dropped** — unchanged. |
| 7. Update comments to list dz/dx, dz/dy | Document new fields | **Revised** — document that gradients are *intentionally omitted* (model learns them) and that orography is normalized by a fixed divisor. |
| 8. Verify channel counts stage by stage | Good practice | **Still valid**, but no change is expected. Source of truth = `expected_in_channels_assimilation` (`train_module.py:90`) and `expected_in_channels_forecast` (`train_module.py:103`) — the hardcoded `4` there stays `4`. |

## Net scope for the encoder static step

1. Repoint `build_elev_vars.py` / `run_build_static.sh` at the OK 2.5 km domain
   (orography + LSM source, OK grid files, output dir).
2. Fix the res-aware output filename in `build_elev_vars.py`.
3. Add fixed-divisor normalization of orography (and, for consistency, consider applying
   the same fixed-divisor convention in the loaders that currently differ — raw at
   `loader.py:121-124`, per-domain z-score at `loader.py:1717-1719`).
4. **No channel-count changes anywhere** — encoder, processor, decoder, e2e all unchanged
   on the static axis.
