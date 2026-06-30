# RTMA OK 15-minute cadence: per-month files + ERA5-type indexing

## Goal

Run the RTMA OK surface encoder (`--obs_set rtma_surface`) at **15-minute** intervals (96
frames/day) instead of daily. Store **both the observation files and the target files
per-month**, and address them by an ERA5-type *(file, row)* scheme. This removes the existing
"6H-vs-daily" binary and the per-modality offset machinery for the surface path.

Scope: the **encoder / assimilation** surface-only path. The global-obs modalities
(ICOADS/IGRA/AMSU/IASI/HIRS/ASCAT/GridSat) are not used here and are out of scope.

## Background: why the current code can't do 15-min

`time_freq` is a hardcoded **binary** ("6H" vs. daily). Frames-per-day is `4 if "6H" else 1`,
and any non-"6H" value falls into the daily branch. Two daily assumptions break at 15-min:

- **ERA5 target memmap frame count**: `d = days * (4 if "6H" else 1)` (`loader.py:663`) -> only
  1 frame/day; 15-min needs 96. With `rtma_ok_sfc` inferring channels from file size, a
  96x-too-small `d` yields a 96x-too-large channel count -> crash/garbage.
- **ERA5 target frame index**: `load_era5_time` uses `doy = date.dayofyear - 1` for non-6H
  (`loader.py:798`) -> reads 1 frame/day, ignoring intra-day position -> wrong target.

(Plus the time-of-day aux loses sub-hourly resolution -- see below.)

## How the current schemes address a date -> a row

- **Observations** (single big file, arbitrary start date): `row = index + offset`, where
  `offset = build_offsets(freq)[modality][start_date]` is the file row of the run's start date
  (`loader_utils_new.py:build_offsets`). This is the only reason the offset machinery exists.
- **ERA5 target** (per-year files starting Jan 1): `era5_sfc[year-start_year][dayofyear-1]` --
  no offset needed, because every year file starts on a known boundary.

## The design: per-month files for BOTH obs and target

Store one file per calendar month, same convention for obs and target. Then any timestamp maps
to *(file, row)* by pure arithmetic -- **no offsets, no `date_list`, no `build_offsets`**:

```python
STEP_MIN       = 15
frames_per_day = 1440 // STEP_MIN                          # 96
frame_in_day   = (date.hour*60 + date.minute) // STEP_MIN  # 0..95
frame_in_month = (date.day - 1) * frames_per_day + frame_in_day
month_file     = files[(date.year, date.month)]
value          = month_file[frame_in_month]
```

The SAME `(date.year, date.month)` + `frame_in_month` reads both the target month file and the
obs month file, so they are **index-aligned by construction** -- which is exactly the job the
per-modality offsets used to do.

**Worked example -- `2019-03-05 13:30`:**
- `frame_in_day   = (13*60+30)//15 = 810//15 = 54`
- `frame_in_month = (5-1)*96 + 54 = 384 + 54 = 438`
- target: `era5_rtma_ok_sfc_1_15min_2019-03.memmap`, row **438**
- obs:    `{var}_vals_2019-03.memmap` (shape `(days_in_month*96, n_stations)`), row **438**

## File layout

| Kind | Per-month file | Per-frame shape | Frames/month |
|---|---|---|---|
| Target | `era5/era5_rtma_ok_sfc_1_15min_<YYYY>-<MM>.memmap` | `(channels=5, nlon, nlat)` | `days_in_month * 96` |
| Surface obs (per var) | `hadisd_processed/{var}_vals_<YYYY>-<MM>.memmap` | `(n_stations,)` | `days_in_month * 96` |

- Station **coordinates stay static** (`{var}_lon/lat/alt_*.npy`, loaded once) -- only the
  *values* are per-month.
- Per-month time-dim is inferred from file size (as `load_era5` already does for `rtma_ok_sfc`),
  then **asserted** (see caveat) -- leap/short months need no special-casing.
- **No `_train`/`_val`/`_test` split tag in the file names.** Monthly files use only the
  `<YYYY>-<MM>` convention; train/val/test are selected purely by `start_date`/`end_date`. This
  matches existing behavior (HadISD already passes `hadisd_mode="train"` for both the train and
  val datasets and splits by date range) and avoids a second indexing system.

## Code changes (encoder / `rtma_surface` path only)

1. **Named shared helpers** (do NOT extend the `if "6H" else daily` binary -- replace it):
   - `parse_time_freq(time_freq) -> (step_minutes, frames_per_day, freq_tag)`
     (`"6H"`->360/4/`"6"`, `"1D"`->1440/1/`"1d"`, `"15min"`->15/96/`"15min"`). Use this
     **everywhere** a frame count or time-row index is computed.
   - `month_frame(date, step_minutes) -> ((year, month), frame_in_month)` -- the single source
     of truth; the SAME result reads both obs and target.
   - `days_in_month(year, month) -> calendar.monthrange(year, month)[1]`.
2. **Target (ERA5)** `loader.py`:
   - `load_era5`: build a dict `{(year, month): memmap}` instead of the per-year list;
     per-month shape `(days_in_month*frames_per_day, channels, nlon, nlat)`.
   - `load_era5_time`: replace `doy` with the `month_frame(date, step_minutes)` lookup.
3. **Surface obs** `loader.py`:
   - `load_hadisd`: open `{(year, month): memmap}` per variable instead of one big file +
     `hadisd_index_offset`. Coords loaded once (unchanged).
   - `get_index`: read `obs_month[(y, m)][frame_in_month]` using the SAME `month_frame` result
     as the target (this is what guarantees obs/target alignment).
   - Drop `self.hadisd_index_offset` and the `build_offsets` dependency on this path.
   - **Bypass `build_offsets` for this path.** `WeatherDataset.__init__` calls
     `self.offsets = build_offsets(self.time_freq)` before modality loading. It does NOT crash on
     `"15min"` (pandas accepts the freq), but for `hadisd` it builds a ~2.5M-timestamp index over
     1950->2021 only to discard it. Gate it, e.g.
     `self.offsets = None if (self.obs_set=="rtma_surface" and self.time_freq=="15min") else build_offsets(self.time_freq)`.
4. **Time-of-day aux** `get_time_aux` (`loader.py:711`): `time_of_day = hour + minute/60`.
   **Backward-compatible**: 6H/1D timestamps have `minute==0`, so this equals `hour` and is
   bit-identical for existing runs; it only adds resolution at sub-hourly cadence.
5. **Config** `grid_config_ok.yaml`: add `{month}` to the era5/obs file-name templates
   (currently `{year}` only); obs and target use the same `<YYYY>-<MM>` convention.
6. **`time_freq`** wiring: accept `"15min"` (pandas `pd.date_range` already supports it); add a
   `15min` freq_tag.

## What this eliminates / simplifies

- `build_offsets`, `generate_offsets`, `date_list`, `offsets[d]`, `hadisd_index_offset` --
  unused on the surface path.
- The `[WARN] start_date not in offsets` constraint disappears -- a run can start on **any**
  date (no need to pre-register start dates in `date_list`).
- The `4 if "6H" else 1` frames-per-day binary is replaced by the `STEP_MIN` formula.

## Open decisions for colleagues

1. **Climatology / 00z background.** DECIDED: the climatology channel carries a **per-actual-day
   00z background** (a forecast valid at 00 UTC), stored **one file per month**. See the
   "Climatology: per-month 00z background" section below for the implementation plan. (Open
   sub-decisions there: raw vs. normalized, file name, background lead time.)
2. **File-name convention.** `_<YYYY>-<MM>` vs. `_<YYYY>_<MM>`; zero-pad month. Must be identical
   for obs and target.
3. **On-demand vs. pre-open.** Pre-open all month memmaps into a dict (lazy mmap, cheap) vs.
   open per access. A multi-year run is dozens of files per variable -- pre-open is fine.

## Alignment caveat (HARD assert, both files)

This only works if obs and target month files share the **exact same timestamp grid**: same
first frame (both start at `00:00` on day 1), same 15-min cadence, same frame count per month.
Since the index->row math is shared, any convention mismatch silently misaligns obs and target.

Inferring the time-dim from file size is convenient but **not sufficient** for alignment --
infer, then **hard-assert at load time, for BOTH the obs and the target month files**:

```python
assert time_dim == days_in_month(year, month) * frames_per_day, (
    f"{path}: {time_dim} frames != expected {days_in_month(year, month)*frames_per_day}"
)
```

A wrong frame count (e.g. a missing/partial month) must fail loudly, not silently shift rows.

## Verification

- Pick a known timestamp; confirm `month_frame` maps it to the expected `(file, row)` and that
  obs and target rows correspond to the same time.
- `[DEBUG] era5_mode ... time_freq 15min era5_frame_shape ...` shows the expected channels.
- For a 6H/1D regression check: `get_time_aux` output is unchanged (minute==0).

## Daily 00z background (per-month) -- fills the encoder's "climatology" slot (PLAN, not implemented)

**Terminology (review):** this field is a **daily 00z background / prior** -- a date-specific
forecast valid at 00 UTC -- **NOT climatology** in Aardvark's original sense (a multi-year
day-of-year mean). To avoid confusion, name it **`background`** in code and comments
(`self.background`, `_load_background_monthly`); it merely *reuses* the model-facing input slot
`task["climatology_current"]` so the model needs no change. Suggested code comment:

> In the rtma_surface + 15min monthly path, the existing `climatology_current` input slot is
> populated by a normalized daily 00z background field -- a date-specific background/prior, NOT a
> multi-year climatology.

For `rtma_surface` + `15min`, this slot carries a **00z background field** (a forecast at a
chosen lead time, valid at 00 UTC), per **actual day**, stored **one file per month**. It is NOT
derived from the target and is NOT a day-of-year mean. Same 5 fields as the target, so the slot's
channel count = 5 and `in_channels` stays 24.

### File layout
- `era5/background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap`, shape `(days_in_month, C, nlon, nlat)` --
  **one frame per day = that day's 00z background**. `C` = 5 (target fields).
- Built from the background/forecast source (data-side, e.g. an adapted `build_climatology.py`
  that emits per-month daily-00z fields). Out of loader scope.

### Index mapping (reuses the monthly machinery)
For a sample at `date` on day `D` of month `M`, year `Y`:
```python
(year, month), _ = month_frame(date, self.step_minutes)        # same (Y, M) as obs/target
background = self.background[(year, month)][date.day - 1, ...]  # 00z of day D, frame = D-1
```
All 96 intraday samples of day `D` read frame `D-1` -> the same 00z background (no intra-day
time dependence). No leakage: the background is a forecast, distinct from the target.

### Loader changes (all gated on `self.monthly`; existing paths untouched)
1. **`__init__`** -- gate the climatology block (`loader.py:222-236`):
   ```python
   if self.monthly:
       self.background = self._load_background_monthly()   # {(Y,M): memmap (days, C, nlon, nlat)}
       self.climatology_channels = <C, inferred from the first month file>
   else:
       ... existing climatology_data.mmap load ...
   ```
2. **New `_load_background_monthly`** (mirror `_load_era5_monthly`, but **daily** -- one frame
   per day, not 96):
   - path `era5/background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap`;
   - shape `(days_in_month, C, nlon, nlat)`, infer `C` once from the first month;
   - **hard-assert** every month (review point 6):
     - `frames == days_in_month`,
     - `C == target channels (== 5)`,
     - spatial dims `== (nlon, nlat)`.
     (i.e. `nbytes == days_in_month * C_target * nlon * nlat * 4`, with `C` cross-checked against
     the target's channel count, not just inferred independently.)
3. **`get_index`** -- add a monthly branch ahead of the existing climatology lookup
   (`loader.py:1036-1039`):
   ```python
   if self.monthly:
       (year, month), _ = month_frame(date, self.step_minutes)
       climatology = self.background[(year, month)][date.day - 1, ...]
   elif self.time_freq == "6H":
       climatology = self.climatology[date.hour // 6, date.dayofyear - 1, ...]
   else:
       climatology = self.climatology[0, date.dayofyear - 1, ...]
   ```
Reuses `_month_keys`, `days_in_month`. No change to `climatology_channels` semantics (still feeds
`expected_in_channels`).

### Normalization (DECIDED) and remaining data choices
- **Normalized with the TARGET mean/std.** The background is fed normalized:
  `norm_background = (background - target_mean) / target_std`, using the **5-channel
  `rtma_ok_sfc` target** mean/std -- **NOT** the HadISD observation normalization (review point 5;
  see `note_climatology_normalization.md`). Preferred: bake this in at **build time** (store the
  already-normalized field; loader unchanged, file self-describing). Alternative: one-line loader
  normalization with `self.means/self.stds` on the monthly path.
- **File name / token:** `background_rtma_ok_sfc_1_<YYYY>-<MM>.memmap` (keep the target/obs
  `<YYYY>-<MM>` convention).
- **Background lead time:** which forecast lead defines the 00z background -- a data-prep choice;
  does not affect the loader.

### Alignment assert
Same discipline as obs/target: assert each background month file has exactly `days_in_month`
frames so a missing/short month fails loudly rather than mis-indexing the day.

## Implementation status (done, untested)

Implemented and gated behind `self.monthly = (obs_set=="rtma_surface" and time_freq=="15min")`
so all existing `all`/6H/1D paths are byte-identical:

- **`loader_utils_new.py`**: `parse_time_freq`, `days_in_month`, `month_frame`, `TIME_FREQ_TABLE`.
- **`loader.py`** (`WeatherDataset`/`WeatherDatasetAssimilation`):
  - `self.monthly` flag; `parse_time_freq` -> `self.step_minutes/frames_per_day/freq_tag`.
  - `build_offsets` bypassed (`self.offsets = None`) and the start-date warning loop guarded.
  - `_load_era5_monthly` (target) + `_load_obs_monthly` (obs) + `_month_keys`/`_era5_month_path`,
    each with a **hard frame-count assert** (`days_in_month * frames_per_day`).
  - `load_era5_time` and `get_index` use `month_frame(date, step_minutes)` for both target and
    obs (same key -> alignment by construction); `hadisd_index_offset` skipped on this path.
  - `get_time_aux`: `hour + minute/60` (all 3 copies; backward-compatible).
  - Guard: `monthly + two_frames` raises `NotImplementedError` (year-boundary path unsupported).
  - Guard: `month_frame` rejects timestamps not aligned to the cadence (e.g. 13:37 at 15-min) --
    prevents a misaligned start/end date silently snapping to the wrong row.
  - Guard: `time_freq=15min` with `obs_set != rtma_surface` raises `ValueError` (scope explicit).
- **Climatology**: no code change -- the existing non-6H branch already reads daily
  `climatology[0, dayofyear-1]`, i.e. the daily-first-cut.

**Not yet run** (no local Python; OK 15-min data not built). Needs HPC validation. Data layout
required: `era5/era5_rtma_ok_sfc_1_15min_<YYYY>-<MM>.memmap`,
`hadisd_processed/{var}_vals_<YYYY>-<MM>.memmap` (+ static `{var}_lon/lat/alt_train.npy`),
5-channel `mean/std_rtma_ok_sfc_1.npy`, and a daily `climatology_data.mmap`.

## Data contract (preprocessing MUST match the loader)

The loader hard-asserts these on load; **preprocessing must produce -- and ideally verify -- the
same** so a mismatch is caught at build time, not just at run time. (Review point 4.)

| Artifact | Path (relative to data_path) | dtype | Per-frame shape | Frames per month |
|---|---|---|---|---|
| Target | `era5/era5_<era5_mode>_1_15min_<YYYY>-<MM>.memmap` | float32 | `(C_target, nlon, nlat)` | `days_in_month * 96` |
| Obs (per var) | `hadisd_processed/{var}_vals_<YYYY>-<MM>.memmap` | float32 | `(n_stations,)` | `days_in_month * 96` |
| 00z background | `era5/background_<era5_mode>_1_<YYYY>-<MM>.memmap` | float32 | `(C_target, nlon, nlat)` | `days_in_month` (1/day) |

Conventions:
- `<MM>` is **zero-padded** (`01`..`12`); `<YYYY>-<MM>` identical for obs, target, background.
- **Frame 0 of each month = day 1, 00:00**; cadence exactly 15 min, ascending, no gaps.
- Target/obs/background **channel order is identical** (the 5 surface fields, same order as obs).
- Obs station **coords are static** single files (`{var}_lon/lat/alt_train.npy`), NOT per-month.
- Background is the **00z field per actual day**, stored **normalized** with the target mean/std.
- `era5_mode = rtma_ok_sfc` (channel count inferred from file size by the loader).

Loader-side checks already enforce the frame counts (`days_in_month * 96` for obs/target;
`days_in_month` for the background) and reject cadence-misaligned timestamps. Mirroring the same
frame-count check in the preprocessing scripts is recommended.

## Related

- `plan_rtma_phase1.md` -- surface-only encoder (`--obs_set rtma_surface`), 5 obs -> 5 fields.
- `rtma_ok_sfc` era5_mode (channel count inferred from file size).
- `grid_config_ok.yaml` -- OK domain grid.
