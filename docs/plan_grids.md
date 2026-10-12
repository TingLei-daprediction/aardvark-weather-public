# Plan: Make Grid Parameters Configurable + Assert Against On-Disk Grid Arrays

Goal: replace hardcoded grid sizes with configurable parameters, and **validate them at
load time against the actual grid/data arrays** so code and data can't silently drift.

## 1. Background (the two grids)

- **Grid A — data/target grid** (currently `240 lon x 121 lat`): observation-aggregation
  grid + ERA5/climatology storage + model output grid. Defined by
  `era5_x_<res>.npy` / `era5_y_<res>.npy`.
- **Grid B — inner ViT grid** (currently `256 x 128`, `int_x`/`int_y`): the encoder
  upsamples to it for the ViT, then downsamples back. Used only on the assimilation path;
  the forecast path runs the ViT directly at Grid A.

## 2. Source of truth vs. derived vs. configured

- **Grid A is *derived*, not configured.** Treat `era5_x_<res>.npy`/`era5_y_<res>.npy` as
  the single source of truth: `nlon = len(era5_x)`, `nlat = len(era5_y)`. Everything else
  is asserted against these.
- **Grid B is *configured*** (`int_x`, `int_y`), with constraints validated (see s5).
- **Domain extent** (lon/lat min/max) is also derived from `era5_x`/`era5_y` -- this fixes
  the regional bug (s6).

## 3. Config parameters to introduce (single place, threaded CLI/config -> model + loaders)

- `nlon`, `nlat` -- derived from grid files, carried explicitly so every consumer reads the
  same value.
- `int_x`, `int_y` -- inner ViT grid (configured; default keeps current 256x128).
- Implicit: domain extent arrays (the grid files themselves).

## 4. Assertion strategy (the core ask)

At dataset/model construction, after loading grid files, assert once, loudly, with names:

1. **Grid files consistent:** `era5_x`/`era5_y` 1-D lengths -> set `nlon`, `nlat`.
2. **Static fields match:** `elev_vars_<res>.npy` spatial dims == `(nlat, nlon)`
   (after the known transpose convention).
3. **Climatology matches:** climatology mmap trailing spatial dims per `data_shapes.py`.
4. **Observation memmaps match:** AMSU/HIRS/IASI/ASCAT/etc. spatial dims.
5. **Norm factors / IC memmaps** spatial dims == Grid A.
6. **Grid B compatibility:** `int_x >= nlon`, `int_y >= nlat`, both even (s5).

Principle: derive `nlon/nlat` from one canonical file, then assert all other arrays against
it -- rather than hardcoding shapes. Prefer inferring shapes from file metadata where
feasible.

## 5. Constraints to validate for Grid B

- `int_x`, `int_y` even (ideally power-of-two) -- odd dims break stride-2/pooling internally.
- `int_x >= nlon`, `int_y >= nlat` (oversample, don't undersample).
- ViT pads for patch size, so exact divisibility by `patch_size` is NOT required
  (today's 256x128 isn't divisible by patch_size=3) -- assert the padding path handles the
  chosen size rather than asserting divisibility.

## 6. Fix the regional-extent bug (bundled in)

`models.py:88-91` builds `int_grid` from a hardcoded global
`linspace(0,360,nlon)`/`linspace(-90,90,nlat)`. Replace with the actual `era5_x`/`era5_y`
(or domain min/max), so the obs-aggregation grid matches the real domain extent. Without
this, a regional domain aggregates observations across the whole globe.
(Naming trap: variable `int_grid` is Grid A; `int_x/int_y` is Grid B.)

## 7. Open decisions

- **One grid or two?** The forecast path already runs the ViT directly on Grid A. For a
  regional domain we could drop Grid B entirely (set ViT `img_size = (nlon, nlat)`, skip the
  up/down resample). Or keep two grids for parity with the global model.
- **Where config lives:** CLI args vs. a domain config file loaded once.
- **`data_shapes.py`:** infer-from-file vs. keep explicit constants per domain.

## 8. Change surface (where literals get replaced)

- **Model:** `int_x/int_y` `models.py:58-59`; `int_grid` `88-91`; ViT `img_size`
  `142`,`153`; interpolate sizes `311`,`323`,`459`,`475`,`479`; `vit.py:104` + pos-embed.
- **Loaders:** gridding dims `loader.py:568-569`,`1249-1250`,`1890-1891`; memmap shapes
  `1085`,`1103`,`1491`,`1653-1676`,`1704`.
- **Data-shape metadata:** `data_shapes.py:5-6, 49-54`.
- **Misc/decoder:** `misc_downscaling_functionality.py:89-90, 150`.
- **Out of scope:** `unet_wrap_padding.py` 128/256/512/1024 (feature-channel widths, NOT
  spatial); channel counts (settled in `plan_static.md`).

## 9. Phasing

1. Add config + derive `nlon/nlat` from grid files; add the assertion block (no behavior
   change at 240x121/256x128 -- pure validation).
2. Replace literals with config values; re-run at current sizes to confirm parity.
3. Switch to OK-domain grid files; assertions catch mismatches.
4. Decide one-grid-vs-two and (if needed) pos-embed handling / retrain.

## 10. Implementation status (done)

Phases 1-2 implemented (defaults keep 240x121 / 256x128):

- **`loader.py`**: added `grid_dims_from_files()` (reads `nlon/nlat` from
  `era5_x_<res>.npy`/`era5_y_<res>.npy`) and `assert_grid_match()`. Every loader now derives
  `self.nlon/self.nlat` and asserts `era5_x/era5_y` and `elev_vars` against them
  (`WeatherDataset` [base/encoder], `WeatherDatasetDownscaling`,
  `ForecasterDatasetDownscaling`, `ForecastLoader`, `AardvarkICDataset`). All `x=240/y=121`
  res-branches and IC/ERA5 memmap-shape literals replaced with `self.nlon/self.nlat`.
  Climatology shape now passes `nlon/nlat`.
- **`models.py`**: `ConvCNPWeather` takes `int_x/int_y` (default 256/128); `nlon/nlat`
  derived from grid files; **`int_grid` now built from `era5_x/era5_y` instead of the global
  `linspace` (fixes the regional-extent bug, s6)**; added Grid B validation (even, `>=` Grid
  A); all `img_size`/interpolate literals replaced with `nlon/nlat` (Grid A) or
  `int_x/int_y` (Grid B).
- **`train_module.py`**: `--int_x/--int_y` args; passed to model; cross-check that the
  model's grid files (`model_data_path/grid_lon_lat`) match the dataset's
  (`data_path/era5`).
- **`misc_downscaling_functionality.py`**: derives `nlon/nlat`; station-output assert uses
  them.
- **`data_shapes.py`**: `get_climatology_shape()` accepts `nlon/nlat`.

### Behavior caveat (verified against the grid files)

The `int_grid` fix means the encoder's observation-aggregation grid now uses the **actual**
`era5_x/era5_y` coordinates rather than `linspace(0,360,240)`/`linspace(-90,90,121)`.
Ordering was verified directly from `data/grid_lon_lat/era5_{x,y}_1.npy` (float64):

- **Latitude (`era5_y`)**: stored **ascending** `-90 -> 90`, step 1.5, 121 pts. This is
  **identical** to the old `linspace(-90,90,121)`. No latitude flip is introduced. (The
  `prep_era5_truth.py` "flip lat to 90->-90" comment is misleading -- the reindex targets the
  ascending `lat_tgt` directly.)
- **Longitude (`era5_x`)**: stored `0 -> 358.5`, step **exactly 1.5**, 240 pts. The old
  `linspace(0,360,240)` had step 360/239 ~= 1.5063 and a final 360 point duplicating 0
  (wraparound). So the new grid is the correct 1.5deg grid; the change is a small longitude
  **correction**, not a flip.

Net: at global 240x121 the aggregation grid is unchanged in latitude and slightly corrected
in longitude. Not bit-identical to prior global runs, but benign and more correct. For a
**regional** domain, deriving `int_grid` from the grid files is required (the old global
`linspace` would smear obs across the whole globe).

### Review round 1 fixes (done)

Colleague review (encoder-focused) addressed:

- **Gridded-obs memmap shapes** (`load_amsua/amsub/ascat/hirs/iasi`, daily/1D path) now derive
  their **spatial** dims from `self.nlat/self.nlon`, preserving each instrument's axis order
  (AMSU-A is `(time, nlat, nlon, ch)`; AMSU-B/ASCAT/HIRS/IASI are `(time, nlon, nlat, ch)`).
  `self.nlon/self.nlat` are now set at the top of `__init__`, before any modality loads.
  (GRIDSAT stays on its own fixed 514x200 grid, read from `sat_x/sat_y.npy` -- not Grid A.)
- **Cross-check now compares actual lon/lat values** (`np.allclose` on the model vs dataset
  grid files), not just `nlon/nlat` lengths -- so same-shape/different-extent domains fail.
- **`assert_grid_match` is now orientation-explicit** (takes an expected `(d0,d1)`): raw
  `elev_vars` checked as `(nlat, nlon)`, post-loader tensors as `(nlon, nlat)`.
- **`int_grid` no longer forced to `.cuda()`** -- stays on CPU; `forward()` moves it to the
  batch device.
- **Grid B validation gated** to `decoder == "vit_assimilation"` (only that path uses
  `int_x/int_y`).

### Domain data files routed through the YAML (done)

Beyond the grid coordinate files, the other grid/domain-dependent files now resolve through
`grid_config` (`data_files` section in the YAML; templates with `{res}`, `{era5_mode}`,
`{freq}`, `{year}`; defaults reproduce the original names). Helpers: `elev_vars_path`,
`norm_mean_path`, `norm_std_path`, `era5_memmap_path`, `lat_weights_path`.

Routed everywhere they appear in the training path:
- **Orography** `elev_vars` -- all loaders.
- **ERA5 norm factors** `mean_<mode>_<res>` / `std_<mode>_<res>` -- all loaders.
- **ERA5 memmaps** `era5_<mode>_<res>_<freq>_<year>` -- all loaders' `load_era5`.
- **Latitude weights** `weights_lat_<res>` -- `loss_functions.py`.

Deliberately NOT routed (out of scope / not grid-dependent):
- Per-instrument obs norms (`mean_amsua.npy`, `mean_hadisd_<var>.npy`, ...) -- no res/grid tag.
- ForecastLoader **diff-norm** variants (`mean_diff_*`, `*_6h`, `*_12h`) -- forecast/processor
  specific; left inline.
- `e2e_model.py` / `generate_initial_condition_single.py` -- separate entry points (like the
  grid files, not wired to the config yet).

So `res` is now purely a filename token consumed inside these templates; the encoder data
path (grid + elev + norm + era5 memmap) is fully YAML-controlled, defaults unchanged.

### Grid-file consistency assertion (done)

`grid_config.assert_grid_files_consistent(model_data_path, data_path, res)` loads the model's
and the dataset's grid coordinate files and asserts they match in **both shape and actual
coordinate values** (`np.allclose`) -- so size mismatches *and* same-shape/different-extent
grids fail. Called in `train_module` for **both** model branches (assimilation/forecast and
downscaling). Not yet wired into `e2e_model.py` / `finetune.py` (separate entry points).

### Review round 2 fix (done)

- **On-grid observation files are now asserted** (`assert_memmap_size`): each daily/1D obs
  loader (`amsua/amsub/ascat/hirs/iasi`) checks `getsize(path) == prod(shape)*4` before the
  `np.memmap`. This catches a wrong-resolution obs file (previously a too-large file was
  silently mapped to a prefix). It validates the **full** shape, so a wrong time/channel
  count fails loudly too. Gated to the 1D path (the 6H native-grid files are out of OK
  scope). Caveat: it cannot detect a same-byte-count lat/lon axis swap -- the memmap stores
  no coordinates, so orientation is enforced only by the shape convention we pass.

### YAML grid config (done)

Free grid parameters now live in a YAML read at startup, with built-in defaults:

- **`aardvark/grid_config.yaml`** holds the **explicit era5_x/era5_y file names**
  (`grid_files.loader_x/_y` read relative to `data_path`; `grid_files.model_x/_y` relative to
  `model_data_path`) and the **inner ViT grid `int_x`/`int_y`** (default 256/128). Names are
  explicit (no `{res}` template, though a literal `{res}` is still honoured if present).
  Derived Grid A params (`nlon/nlat`, extent) are deliberately NOT in the YAML -- they come
  from the files. The loader (`data_path/era5/...`) and model (`model_data_path/grid_lon_lat/...`)
  read the *same* grid from two roots; the train_module cross-check guards against drift.
- **`aardvark/grid_config.py`**: `load_grid_config(path)` (merges YAML over defaults),
  `set_active_config()` (process-wide), and path helpers
  `loader_grid_{x,y}_path` / `model_grid_{x,y}_path`. If no config is installed, helpers fall
  back to defaults = the original hard-wired names (backward compatible).
- **`train_module.py`**: `--grid_config` (defaults to the bundled YAML); installs it per DDP
  process before datasets/model are built; `int_x/int_y` come from the YAML unless `--int_x`/
  `--int_y` override. All grid-file reads in `loader.py`, `models.py`,
  `misc_downscaling_functionality.py`, and the cross-check now go through the helpers.
- **Note**: `e2e_model.py` / `finetune.py` build models without installing a config, so they
  use the defaults (256/128 and the conventional file names). Wire `--grid_config` there too
  if/when running those on the OK domain.

### `res` removed as a variable (safest fix for the divergence risk)

To eliminate the grid-vs-data filename divergence risk entirely, `res` was removed as a
variable from the **model runtime path**:

- **Config templates are now fully explicit** (no `{res}`): grid files `*_1.npy`, and
  `data_files` bake the `_1` (e.g. `mean_{era5_mode}_1.npy`, `era5_{era5_mode}_1_{freq}_{year}.memmap`).
  `{era5_mode}`/`{freq}`/`{year}` remain.
- **`grid_config.py` helpers drop the `res` argument** (`grid_dims_from_files(data_path)`,
  `loader_grid_x_path(data_path)`, `elev_vars_path(data_path)`, `norm_mean_path(root, era5_mode)`,
  `era5_memmap_path(data_path, era5_mode, freq, year)`, `lat_weights_path(aux)`,
  `assert_grid_files_consistent(model_data_path, data_path)`).
- **`--res` CLI arg removed** from `train_module.py`, `e2e_train.py`, `finetune.py`; `self.res`
  and the `res=` constructor params removed from all datasets (`loader.py`), `ConvCNPWeather`
  (`models.py`), `ConvCNPWeatherOnToOff` (`misc_downscaling_functionality.py`), and both
  weighted losses (`loss_functions.py`); `res`/`forecast_config["res"]` reads removed from
  `e2e_model.py`, `generate_initial_condition_single.py`; `--res 1` removed from
  `training/e2e_train_4u_sfc.sh`. ForecastLoader's inline diff-norm names now bake `_1`.
- Verified by grep: no `res` variable remains in `aardvark/*.py` (only a doc comment in the
  YAML). For OK, edit the explicit filenames in the YAML; there is no resolution knob.
- **Out of scope (still use their own `--res`):** the data-generation scripts
  (`build_climatology.py`, `compute_era5_diff_stats.py`, `make_era5_daily_from_6h.py`) and
  their `run_*.sh` wrappers -- these are the pending aux-script work, separate from the model
  runtime.

### Review round 3 (colleague) -- fix + clarifications

- **Fixed:** `grid_config.py` path helpers now use `os.path.join(root, name)` instead of
  `root + name`, so a `data_path`/`model_data_path` without a trailing separator works.
- **Clarification:** "all loader memmap spatial dims derive from nlon/nlat" applies to the
  encoder's **daily Grid-A gridded obs** only. **Station obs (HadISD/ICOADS/IGRA), GridSat,
  and legacy 6-hour native obs keep their own source-native shapes** (different grids; not
  Grid A).
- **Caution (res vs explicit filenames):** grid files are now explicit (`*_1.npy`) while the
  `data_files` templates still use `{res}`. If someone changes `--res` without updating the
  grid filenames, grid and data files can diverge. The shape assertions catch most cases
  (elev_vars, era5 memmap), but **norm factors are per-channel and would NOT be caught**.
  Mitigation: keep a single consistent filename label, or use `{res}` consistently in the
  YAML. (The clean end state is the pending `res` removal -- make `data_files` explicit too.)

### Remaining (Phase 3 / data-pipeline, not done)

- **Data-generation / auxiliary scripts** still hardcode `era5_x_1.npy`, `era5_y_1.npy`,
  `elev_vars_1.npy`, and 240x121 (e.g. `build_elev_vars.py`, climatology/obs/ERA5-truth prep).
  Outside the encoder runtime path, but must be made configurable before OK-domain data is
  generated end-to-end.

- **Obs memmap TIME and CHANNEL dims** in `data_shapes.py` are still constants (only the
  spatial dims are now grid-derived). For OK obs memmaps with a different frame count, infer
  the time dim from file size (as `load_era5` does) when that data exists. Until then,
  `assert_memmap_size` will flag any such mismatch loudly rather than misread silently.
- **ViT positional embeddings** when actually changing `int_x/int_y` or `nlon/nlat`
  (`vit.interpolate_pos_embed`) -- only needed when sizes change / reusing weights.
- **One-grid-vs-two decision** (s7) still open.
