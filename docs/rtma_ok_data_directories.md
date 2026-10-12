# Aardvark RTMA-OK Hourly Encoder: Data Directory Specification

**Project:** AI RTMA-OK surface analysis (Aardvark encoder, Oklahoma 2.5 km domain)
**Configuration:** `--obs_set rtma_surface`, `--era5_mode rtma_ok_sfc`, `--time_freq 1H`, grid
configuration `aardvark/grid_config_ok.yaml`
**Date:** 2026-07-11

## 1. Overview

The training system reads data from three directory roots, supplied on the command line of
`train_module.py`. All file names below are resolved through the templates in
`grid_config_ok.yaml`, which adopts the "urma" naming token for the Oklahoma dataset (the
source is URMA, not ERA5).

1. **`--data_path`** — the primary data tree. Holds all gridded truth/background data and all
   observation data. This is where the preprocessing pipeline writes its outputs.
2. **`--aux_data_path`** — the root under which `norm_factors/` is found. In our setup this is
   the same directory as `data_path`.
3. **`--model_data_path`** — the model-side root. For the encoder stage it supplies exactly two
   files: the model's copy of the grid coordinate axes. Conventionally this is the repository's
   `data/` directory.

At startup, the system verifies that the grid coordinate files under `model_data_path` and
`data_path` are identical in shape and values (`assert_grid_files_consistent`), and validates
every gridded file against the grid dimensions, including an orientation check of the static
elevation file. Any inconsistency terminates the run with an explicit error.

## 2. Grid definition

The Oklahoma grid is a regular latitude-longitude grid with 331 longitudes (256.800 E to
265.710 E, step 0.027 deg) and 171 latitudes (33.500 N to 37.325 N, step 0.0225 deg). All files
follow the project convention: longitude ascending west-to-east on the 0-360 branch, latitude
ascending south-to-north (southwest origin). Gridded arrays are stored with longitude before
latitude, i.e. `(..., nlon, nlat)`; the single exception is the static elevation file, stored
`(4, nlat, nlon)`.

## 3. Contents of `data_path`

### 3.1 `urma/` — gridded truth, background, and grid axes (GENERATED)

Produced by `scripts/prep_urma_ok_hourly.py` and `scripts/normalize_background.py` (submitted
together via `scripts/run_prep_urma_ok_hourly.sbatch`) from the URMA grib2 archive, with no
interpolation; values are copied as decoded.

- `urma_rtma_ok_sfc_1_1h_<YYYY>-<MM>.memmap` — hourly analysis targets, one file per month,
  float32, shape `(days_in_month * 24, 5, 331, 171)`, raw physical units. Channel order:
  2 m temperature [K], 2 m specific humidity [kg/kg], surface pressure [Pa], 10 m u-wind [m/s],
  10 m v-wind [m/s]. The loader normalizes targets at runtime.
- `background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap` — daily 00 UTC first-guess (background/prior),
  one file per month, shape `(days_in_month, 5, 331, 171)`, stored **normalized with the target
  mean/std** (build-time normalization). This field feeds the encoder's background input slot;
  all 24 hourly samples of a day share that day's 00 UTC background. The file name carries no
  cadence tag because one frame per day is cadence-independent (the same files would serve a
  15-minute run).
- `background_raw_rtma_ok_sfc_1_<YYYY>-<MM>.memmap` — un-normalized intermediate of the above;
  reproducible from the gribs and deletable, but convenient to keep in case the normalization
  factors are ever recomputed.
- `urma_x_ok.npy`, `urma_y_ok.npy` — the 1-D grid coordinate axes (longitude, latitude; float64
  degrees, ascending), derived from the grib metadata and written as a side product.
- `elev_vars_ok.npy` — **PENDING.** Static fields: orography [m], land-sea mask, sin(latitude),
  cos(latitude); shape `(4, 171, 331)`, southwest origin. To be produced by
  `scripts/build_elev_vars.py --tag ok --name_root urma` once an orography/land-sea-mask source
  is chosen (ERA5 static file via nearest-neighbor, or URMA terrain height if available in the
  gribs).

### 3.2 `hadisd_processed/` — surface observations (PENDING)

The observation side of the encoder; conversion tooling not yet written. Required, per variable
`tas, sh, psl, u, v`:

- `{var}_vals_1h_<YYYY>-<MM>.memmap` — station observation values, one file per month, shape
  `(days_in_month * 24, n_stations)`, aligned frame-for-frame with the target months
  (frame 0 = day 1, 00 UTC).
- `{var}_lon_train-<YYYY>-<MM>.npy`, `{var}_lat_train-<YYYY>-<MM>.npy`, and
  `{var}_alt_train-<YYYY>-<MM>.npy` — month-specific station coordinates (longitude on the
  0-360 branch). Station membership, ordering, and count may change by month; each coordinate
  set must match the corresponding monthly values-file columns exactly.

Physical conventions must match the target channels: pressure observations are station surface
pressure in Pa (not mean-sea-level pressure), humidity is specific humidity, and winds are true
u/v components.

## 4. Contents of `aux_data_path` (same directory as `data_path`)

### `norm_factors/`

- `mean_rtma_ok_sfc_1.npy`, `std_rtma_ok_sfc_1.npy` — **GENERATED.** Per-channel mean/std of
  the target over the training period, shape `(5,)`. Used at every training/evaluation step to
  normalize targets, at inference to un-normalize predictions, and at build time to normalize
  the background. Note: these and the normalized background files form a matched set; if the
  norms are recomputed, the backgrounds must be re-normalized from the raw files.
- `mean_background_rtma_ok_sfc_1.npy`, `std_background_rtma_ok_sfc_1.npy` — **GENERATED.**
  Per-channel statistics of the raw background, written by `normalize_background.py`.
  Diagnostic only (first-guess bias assessment); not read by the training system.
- `mean_hadisd_{var}.npy`, `std_hadisd_{var}.npy` — existing shared scalar factors used
  when `--obs_norm_mode static` (the backward-compatible default).
- `mean_hadisd_{var}_train-<YYYY>-<MM>.npy` and
  `std_hadisd_{var}_train-<YYYY>-<MM>.npy` — station-dependent monthly vectors used when
  `--obs_norm_mode monthly`. Their shape is that variable/month's station count and their
  ordering must match the corresponding monthly lon/lat/alt and values files. They are written
  by `scripts/stage_ok_obs_month.py` from the same monthly values matrix.

## 5. Contents of `model_data_path`

For the encoder stage, exactly two files, both copies of the axes in section 3.1:

- `grid_lon_lat/urma_x_ok.npy`
- `grid_lon_lat/urma_y_ok.npy`

The model derives its observation-aggregation grid and positional encoding channels from these;
the inner transformer grid size (384 x 192) comes from `grid_config_ok.yaml`, not from data
files. The startup consistency check compares these copies against the `data_path` copies
value-by-value. Setup is a one-time copy after preprocessing:

```
cp <data_path>/urma/urma_x_ok.npy <data_path>/urma/urma_y_ok.npy <model_data_path>/grid_lon_lat/
```

(The repository's `data/` directory also contains global-domain files used by the demo
notebooks; they are unrelated to, and can coexist with, the Oklahoma files.)

## 6. Status summary

| Item | Location | Status |
|---|---|---|
| Hourly targets, 2022-01 to 2022-06 | `data_path/urma/` | Generated |
| Daily 00 UTC backgrounds (normalized + raw) | `data_path/urma/` | Generated |
| Grid axes (both copies) | `data_path/urma/`, `model_data_path/grid_lon_lat/` | Generated / one-time copy |
| Target norm factors | `norm_factors/` | Generated |
| Background statistics (diagnostic) | `norm_factors/` | Generated |
| Static elevation file | `data_path/urma/` | Pending (needs orography/LSM source) |
| Hourly observation files + station coordinates | `data_path/hadisd_processed/` | Pending (tooling to be written) |
| Observation norm factors | `norm_factors/` | Pending (with observations) |

Once the two pending items exist, constructing the training dataset for a single month serves
as the end-to-end validation: the loader hard-asserts every file's frame count, grid shape,
channel consistency, and static-file orientation at startup.
