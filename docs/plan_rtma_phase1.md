# RTMA Phase 1: surface-only encoder (5 obs -> 5 surface fields)

## Goal

A configurable RTMA surface-only assimilation mode: the encoder ingests **only the 5 RTMA
surface observations** (`tas, sh, psl, u, v` -- note **specific humidity `sh`**, not dewpoint)
and predicts **5 corresponding surface target fields** (`t2m, q2m, sp/msl, u10, v10`) on the OK
2.5 km grid. All other observation modalities (IASI, ASCAT, ICOADS, GridSat, AMSU-A/B, IGRA,
HIRS) are switched off. Minimal changes to the existing code; gated behind one new option
(`--obs_set rtma_surface`) so the global path is untouched.

### Naming and the `sh` vs `tds` difference (important)

This is **not** literally "HadISD-only." HadISD's variable set is `[tas, tds, psl, u, v]`
(dewpoint); RTMA Phase 1 uses `[tas, sh, psl, u, v]` (**specific humidity**). The humidity
variable differs on **both** sides:

- **Obs:** `sh` comes from RTMA surface obs (or is derived from station dewpoint + pressure
  upstream), not the HadISD `tds` files.
- **Target:** the matching field is **2 m specific humidity `q2m`**, which ERA5 does NOT store
  directly (`prep_era5_truth.py` `name_map` has only `specific_humidity -> q`, a 3-D
  pressure-level field, and `2m_dewpoint_temperature -> d2m`). `q2m` must be **derived** (from
  `d2m` + surface pressure) or taken from a surface/lowest-model-level source. Real data task.

The encoder machinery is **variable-name-agnostic** (`encoder_hadisd` just loops set-convs over
whatever per-variable `(lon, lat, vals)` arrays it is given), so two wiring options:

- **(A) Reuse the HadISD per-variable path** with RTMA files and the `sh` variable swapped in
  for `tds` -- least code, but uses the `hadisd_*` plumbing/naming.
- **(B) Dedicated `rtma_surface` modality** (own loader method, set-convs, encoder) -- cleaner
  separation, slightly more code; mirrors the "add a modality" recipe in `plan_obs.md`.

Recommendation: **(A)** for Phase 1 speed; refactor to (B) if the conflation with HadISD naming
becomes confusing.

## What ALREADY works (no code change)

- **Output side is already configurable.** `out_channels = end_ind - start_ind`
  (`train_module.py:362`), the target is sliced `era5_target[..., var_start:var_end]`
  (`loader.py:988-989`), and the model returns exactly `out_channels`. So predicting **5
  contiguous target channels** is just `--start_ind 0 --end_ind 5 --out_channels 5`.
- **Loss is clean.** `--loss rmse` -> `RmseLoss(start_ind=0, end_ind=end_ind-start_ind)`
  (`train_module.py:156-161`): plain RMSE over the 5 predicted channels, **no per-variable
  weight file required** (`--weight_per_variable` is ignored by `RmseLoss`).
- **HadISD per-variable encoding** already grids each variable independently onto Grid A
  (all 5 now used after the `plan_hadisd_v_drop.md` fix).
- **OK grid** is already plumbed via `grid_config_ok.yaml`.

So Phase 1 is mostly about **turning the other obs OFF** on the input side, plus a **5-channel
surface target memmap** on the data side.

## What needs ADDING

### A. Obs selection -- the one new option (code)

Add a single flag, e.g. `--obs_set {all, rtma_surface}` (default `all` = current behavior).
Thread it through the three layers that currently hard-code the modality set (the same three
identified in the configurable-obs discussion):

1. **Loader** (`loader.py:133-158` + `get_index`): when `rtma_surface`, **skip loading and
   task-dict packaging of all non-surface modalities** and load the RTMA surface obs
   (`tas, sh, psl, u, v`) via the per-variable path. Skipping is REQUIRED, not just an
   optimization -- the OK domain has no global AMSU/IASI/GridSat files, so the loader would
   crash trying to memmap them. Keep: the surface obs, plus the aux blocks (elev, climatology,
   time) which are not modality-gated today.
2. **Model.forward** (`models.py:378-400`): build `encodings = [encoder_hadisd, elev,
   climatology, time]` only. (Store `self.obs_set` from the constructor.)
3. **`expected_in_channels_assimilation`** (`train_module.py:75-100`): a `surface` branch =
   `2*5 (hadisd) + aux_total`, where `aux_total = 4 (elev) + climatology_channels + 5 (time)`.
   Let `--in_channels` auto-derive (omit the flag) so it can't drift.

**Guardrail (closes the main implementation risk).** The model already asserts
`in_channels == expected_in_channels` at construction (`models.py:47-55`), so a wrong
`--in_channels` fails fast -- but that only checks *config vs. formula*, NOT *formula vs. the
actual concatenated tensor*. To catch a `surface`-branch disagreement between
`train_module.py` (the formula) and `models.forward` (the real `encodings` list), add a one-line
named assertion right after `x = torch.cat(encodings, dim=1)` (`models.py:466`):
`assert x.shape[1] == self.in_channels` with a clear message. This turns an otherwise cryptic
ViT shape error into an immediate, named failure -- directly de-risking the three-layer
consistency concern.

### Channel count (confirmed against code)

| Block | Channels | Source |
|---|---|---|
| HadISD (5 vars x 2) | 10 | `encoder_hadisd` after the `v` fix |
| elev / static | 4 | `era5_elev` (orography, LSM, sin/cos lat) |
| climatology | 5 | sfc5 climatology, auto-inferred from file size |
| time | 5 | `get_time_aux` = cos/sin(doy), cos/sin(tod), year (`loader.py:712-720`) |
| **total (with climatology)** | **24** | auto-derived |
| total (climatology removed) | 19 | requires the extra gate below |

Implementation note: this is the minimal version of the **registry/`obs_modalities`** refactor.
For Phase 1 a binary flag is enough; structurally it generalizes the existing `--disable_igra`
toggle to "disable everything except the surface obs."

### Data is ASSUMED GIVEN (out of scope -- no pre-processing-script changes)

Per the scope decision, the pre-processing scripts (`prep_era5_truth.py`,
`build_climatology.py`, the obs converters) are **left untouched**. We assume the data side
provides, on the OK grid and with **consistent variable order** across all three:

- **Obs samples:** the 5 RTMA surface obs `tas, sh, psl, u, v` (per-variable
  `(lon, lat, vals)`), in the same per-variable format the surface encoder already consumes.
- **Target samples:** a 5-channel surface target memmap, channels ordered to match the obs
  (`t2m, q2m, sp/msl, u10, v10`), with its norm factors.
- **Climatology samples:** a matching 5-channel climatology memmap.

Because the variable order is assumed consistent across obs / target / climatology, the run just
selects the target block with the **already-configurable** output args:
`--start_ind 0 --end_ind 5 --out_channels 5`. No code change on the output/target side.

**era5_mode = `rtma_ok_sfc` (implemented).** A clean dedicated mode was added rather than
overloading `4u_sfc`: `load_era5` now infers the channel count from file size for `rtma_ok_sfc`
(all 3 `load_era5` variants in `loader.py`), and it is in the `--era5_mode` choices in
`train_module.py`. OK target/norm/era5 files use this label via the `grid_config_ok.yaml`
templates (`era5_rtma_ok_sfc_1_1d_<year>.memmap`, `mean_rtma_ok_sfc_1.npy`, etc.). Only the
encoder path is wired; the forecast/downscaling loaders' `30 if 4u_sfc else 24` channel-count
lines are intentionally left untouched (out of Phase 1 scope).

### Climatology stays IN (24 channels)

Climatology is concatenated **unconditionally** in `forward` (`models.py:389`), so Phase 1
**keeps it** -- 24 channels total. Dropping it (the 19-channel pure-obs variant) would require a
*fourth* gated block (forward + task dict + channel-count formula) and is deferred until the
24-channel path works. With the data assumed consistent, the climatology block is just the
5-channel context input alongside the obs.

## Change surface (summary)

| File | Change | Kind |
|---|---|---|
| `train_module.py` | add `--obs_set {all, rtma_surface}`; `rtma_surface` branch in `expected_in_channels_assimilation`; pass `obs_set` to dataset + model | code |
| `loader.py` | gate non-surface loads + task keys on `obs_set`; load the 5 RTMA surface vars via the per-variable path; (optional 1-line) infer `levels` from file size for the target mode | code |
| `models.py` | store `self.obs_set`; build `encodings` conditionally; add the post-concat channel assertion | code |

**Code only.** No changes to set_convs, ViT, UNet, the output/loss path, or any
**pre-processing script** (`prep_era5_truth.py`, `build_climatology.py`, obs converters are
untouched). Data is assumed prepared and consistent.

## Open decisions for colleagues (code/config only)

Variable alignment across obs / target / climatology (pressure pairing, units, `q2m`/humidity,
`ws/wd -> u/v`) is **assumed handled by the data side** and is out of scope here. Remaining
code/config choices:

1. **Wiring of `rtma_surface`.** Reuse the existing HadISD per-variable path (option A, least
   code) vs. a dedicated `rtma_surface` modality (option B, cleaner). Recommend A for Phase 1.
2. **Month-varying station locations.** The RTMA monthly path loads one coordinate set per
   variable/month and NaN-pads to a collatable size. This supports evolving membership/order/count.
3. **Length scale.** Set a small `--cmd_init_ls` for the 2.5 km grid (default ~0.36 deg
   over-smooths). See `plan_obs.md`.
4. **Grid B.** On the regional grid, decide ViT-at-Grid-A vs keep inner grid (`plan_grids.md`
   s7); `int_x/int_y` already raised in `grid_config_ok.yaml`.

(The pure-obs / 19-channel variant -- dropping climatology -- is deferred, per "Climatology
stays IN" above.)

## Verification

- `[DEBUG] encodings[i] shape:` (`models.py:401-404`) should list only the surface-obs block
  (+ elev, climatology, time) -- no IASI/AMSU/etc.
- Auto-derived `--in_channels` matches the `[INFO] assimilation input channels` line.
- `out_channels == 5`; target slice `[0:5]`; loss runs over 5 channels.

## Related

- `plan_hadisd_v_drop.md` -- the `v` fix (prerequisite; all 5 HadISD vars now encoded).
- configurable-obs discussion -- `--obs_set` is the minimal slice of that refactor.
- `plan_grids.md` / `grid_config_ok.yaml` -- OK domain grid.
- `plan_obs.md` -- off-grid obs contract, length scale, static vs per-sample.
