# RTMA OK sub-daily cadence: per-month files + ERA5-type indexing

> **Status (2026-07-24, verified against code).** This started as a 15-minute-specific plan; the
> implementation that landed is **cadence-generic**. The per-month path is selected by
> `obs_set == "rtma_surface"` **and** `time_freq in ("15min", "1H")`, and every downstream site
> reads `step_minutes` / `frames_per_day` / `freq_tag` from a single table rather than a literal.
>
> | Cadence | Status |
> |---|---|
> | **`1H`** (24 frames/day) | **In production.** Jan 2022 OK surface encoder, 2xH100 DDP. |
> | **`15min`** (96 frames/day) | **Code path complete and shared with 1H; no data built, never run.** |
>
> Because both cadences traverse the *same* code, 1H running correctly is strong evidence the
> 15-min path is correct **up to the data**. The remaining 15-min work is entirely
> **preprocessing-side** -- see "Remaining work for 15-minute" below.

## Goal

Run the RTMA OK surface encoder (`--obs_set rtma_surface`) at sub-daily cadence instead of
daily. Store **both the observation files and the target files per-month**, and address them by
an ERA5-type *(file, row)* scheme. This removes the old "6H-vs-daily" binary and the
per-modality offset machinery for the surface path.

Scope: the **encoder / assimilation** surface-only path. The global-obs modalities
(ICOADS/IGRA/AMSU/IASI/HIRS/ASCAT/GridSat) are not used here and are out of scope.

## Background: why the original code could not do sub-daily

`time_freq` was a hardcoded **binary** ("6H" vs. daily). Frames-per-day was `4 if "6H" else 1`,
and any other value fell into the daily branch. Two daily assumptions broke:

- **ERA5 target memmap frame count**: `d = days * (4 if "6H" else 1)` -> only 1 frame/day. With
  `rtma_ok_sfc` inferring channels from file size, a too-small `d` yields a proportionally
  too-large channel count -> crash/garbage.
- **ERA5 target frame index**: `load_era5_time` used `doy = date.dayofyear - 1` for non-6H ->
  read 1 frame/day, ignoring intra-day position -> wrong target.

## Cadence table -- the single source of truth

`loader_utils_new.py:12-17`. **Extend here to add a cadence**; nothing else hardcodes a rate.

```python
TIME_FREQ_TABLE = {          # time_freq -> (step_minutes, frames_per_day, freq_tag)
    "6H":    (360,  4,  "6"),
    "1D":    (1440, 1,  "1d"),
    "1H":    (60,   24, "1h"),
    "15min": (15,   96, "15min"),
}
```

`parse_time_freq` (`loader_utils_new.py:20-27`) raises `ValueError` naming the supported set for
anything else, so a typo fails at construction rather than silently taking the daily branch.
`--time_freq` itself is a free-form string (`train_module.py:500`, default `1D`); validation is
`parse_time_freq`'s job.

## Index mapping

`month_frame` (`loader_utils_new.py:35-57`) maps a timestamp to *(file, row)* by pure
arithmetic -- **no offsets, no `date_list`, no `build_offsets`**:

```python
frames_per_day = 1440 // step_minutes
frame_in_day   = (date.hour*60 + date.minute) // step_minutes
frame_in_month = (date.day - 1) * frames_per_day + frame_in_day
```

The **same** `(year, month)` + `frame_in_month` reads the target month file and the obs month
file, so they are **index-aligned by construction** -- exactly the job the per-modality offsets
used to do. `days_in_month` (`loader_utils_new.py:30-32`) uses `calendar.monthrange`, so
leap/short months need no special-casing.

**Worked example -- `2019-03-05 13:30`:**

| Cadence | `frame_in_day` | `frame_in_month` |
|---|---|---|
| `15min` | `810 // 15 = 54` | `(5-1)*96 + 54 = 438` |
| `1H` | `810 // 60` -> rejected (see guard below) | -- |
| `1H`, at `13:00` | `780 // 60 = 13` | `(5-1)*24 + 13 = 109` |

## File layout

| Kind | Per-month file | Per-frame shape | Frames/month |
|---|---|---|---|
| Target | `urma/urma_{era5_mode}_1_{freq_tag}_<YYYY>-<MM>.memmap` | `(C, nlon, nlat)` | `days_in_month * frames_per_day` |
| Surface obs (per var) | `hadisd_processed/{var}_vals_{freq_tag}_<YYYY>-<MM>.memmap` | `(n_stations,)` | `days_in_month * frames_per_day` |
| 00z background | `urma/background_{era5_mode}_1_<YYYY>-<MM>.memmap` | `(C, nlon, nlat)` | `days_in_month` (1/day) |

Naming rules (templates: `grid_config.py:50-51`, OK overrides `grid_config_ok.yaml:41-42`;
helpers `era5_month_path` / `background_month_path`, `grid_config.py:148-155`):

- **Target and obs carry `freq_tag`** -> a 1-hour and a 15-minute dataset **can coexist in one
  `data_path`** without collision.
- **The background carries NO cadence tag** -- one 00z frame per day is cadence-independent, so
  **the background files already built for the 1H run serve a 15-minute run unchanged.**
- Station coordinates and values are both per-month. Coordinate names are
  `{var}_{lon,lat,alt}_train-<YYYY>-<MM>.npy`, allowing station membership/order/count to evolve.
- **No `_train`/`_val`/`_test` split tag.** Train/val/test are selected purely by
  `start_date`/`end_date`, matching existing behavior (`hadisd_mode="train"` for both datasets).

## Implementation (verified against code)

- **`loader_utils_new.py`**: `TIME_FREQ_TABLE`, `parse_time_freq`, `days_in_month`, `month_frame`.
- **`loader.py`** (`WeatherDataset` / `WeatherDatasetAssimilation`):
  - `self.monthly` gate + `step_minutes/frames_per_day/freq_tag` (`:118-119`).
  - `build_offsets` bypassed (`self.offsets = None`, `:127`) and the start-date warning loop
    guarded (`:148-154`).
  - `_month_keys` / `_era5_month_path` (`:629-637`), `_load_era5_monthly` (`:639-675`),
    `_load_background_monthly` (`:681-724`), `_load_obs_monthly` (`:726-750`) -- each with a
    **hard frame-count assert**.
  - `load_hadisd` monthly branch (`:778-783`); coords unchanged.
  - `load_era5_time` (`:976-978`) and `get_index` (`:1091-1097`) use the same
    `month_frame(date, step_minutes)` -> alignment by construction.
  - Background lookup `self.background[(date.year, date.month)][date.day - 1]` (`:1113-1116`):
    all intraday samples of a day read that day's 00z frame.
  - `get_time_aux` (`:893-914`): `time_of_day = hour + minute/60` -- **backward compatible**
    (6H/1D have `minute == 0`, so this equals the integer hour and is bit-identical); it only
    adds resolution at sub-hourly cadence. Interannual channel `year = 0.0` on the monthly path;
    **channel count stays 5**, so `in_channels` is unchanged.

### Guards (all verified present)

| Guard | Location | Behavior |
|---|---|---|
| Unsupported `time_freq` | `loader_utils_new.py:24-27` | `ValueError` listing the supported set |
| Sub-daily cadence with `obs_set != rtma_surface` | `loader.py:122-126` | `ValueError` -- only the surface path has a per-month layout |
| Timestamp not aligned to cadence (e.g. `13:37` at 15-min) | `loader_utils_new.py:48-53` | `ValueError` -- integer division would silently snap to the wrong row |
| `monthly` + `two_frames` | `loader.py:964-968`, `train_module.py:125-126` | `NotImplementedError` / `ValueError` -- the year-boundary path assumes per-year memmaps |
| Wrong frame count, target/obs | `loader.py:662-667`, `:742-746` | `AssertionError` -- a missing/partial month must fail loudly, not shift rows |
| Background not 1 frame/day, or channels != target | `loader.py:707-716` | `AssertionError` |

## What this eliminated

`build_offsets`, `generate_offsets`, `date_list`, `offsets[d]`, `hadisd_index_offset` -- all
unused on the surface path. The `[WARN] start_date not in offsets` constraint is gone, so a run
can start on **any** date. The `4 if "6H" else 1` binary is replaced by the `STEP_MIN` formula.

## Daily 00z background -- fills the encoder's "climatology" slot

**Terminology:** this field is a **daily 00z background / prior** -- a date-specific forecast
valid at 00 UTC -- **NOT climatology** in Aardvark's original sense (a multi-year day-of-year
mean). It is named `background` in code (`self.background`, `_load_background_monthly`); it
merely *reuses* the model-facing input slot `task["climatology_current"]` so the model needs no
change. Same 5 fields as the target, so the slot's channel count = 5 and `in_channels` stays 24.

**Normalization: normalized with the TARGET mean/std**, baked in at **build time**
(`background = (raw - target_mean) / target_std`), so the loader feeds it as-is and the file is
self-describing. **Not** the HadISD observation normalization. Rationale in
`note_climatology_normalization.md` -- the existing global path feeds climatology *raw* while
obs and target are normalized, a latent suboptimality this path deliberately does not inherit.

**No leakage:** the background is a first-guess forecast, distinct from the target.

## Data contract (preprocessing MUST match the loader)

The loader hard-asserts these on load; preprocessing should verify the same so a mismatch is
caught at build time. `F` = `frames_per_day` (24 for `1H`, 96 for `15min`).

| Artifact | Path (relative to `data_path`) | dtype | Per-frame shape | Frames per month |
|---|---|---|---|---|
| Target | `urma/urma_<era5_mode>_1_<freq_tag>_<YYYY>-<MM>.memmap` | float32 | `(C_target, nlon, nlat)` | `days_in_month * F` |
| Obs (per var) | `hadisd_processed/{var}_vals_<freq_tag>_<YYYY>-<MM>.memmap` | float32 | `(n_stations,)` | `days_in_month * F` |
| 00z background | `urma/background_<era5_mode>_1_<YYYY>-<MM>.memmap` | float32 | `(C_target, nlon, nlat)` | `days_in_month` (1/day) |

Conventions:

- `<MM>` is **zero-padded** (`01`..`12`); `<YYYY>-<MM>` identical for obs, target, background.
- **Frame 0 of each month = day 1, 00:00**; cadence exact, ascending, no gaps.
- Target/obs/background **channel order is identical** (the 5 surface fields, same order as obs).
- Obs station coordinates are per-month and must match the same month's values columns exactly.
- Background is the **00z field per actual day**, stored **normalized** with the target mean/std.
- `era5_mode = rtma_ok_sfc` (channel count inferred from file size by the loader).

### Alignment caveat

This works only if obs and target month files share the **exact same timestamp grid**: same
first frame, same cadence, same frame count per month. Since the index->row math is shared, any
convention mismatch silently misaligns obs and target. Inferring the time dim from file size is
convenient but **not sufficient** -- infer, then **hard-assert**, for BOTH files. The loader
does this (`loader.py:662-667`, `:742-746`).

## Remaining work for 15-minute (all preprocessing-side)

The loader/model need **no changes**. What is missing:

1. **A 15-minute target source.** `scripts/prep_urma_ok_hourly.py` hardcodes `FREQ_TAG = "1h"`,
   `FRAMES_PER_DAY = 24` (`:75-76`) and the analysis filename pattern
   `urma2p5.t{hour:02d}z.2dvaranl_OK.grb2`, which has an hour placeholder only. Generalizing it
   needs a `--step_minutes`/`--freq_tag` pair and a sub-hourly filename pattern.
   **Note:** URMA is an **hourly** product. A genuine 15-minute target requires **RTMA**
   (which does run at 15-minute cadence), not URMA -- this is a data-source decision, not a
   naming change.
2. **15-minute observation files** `{var}_vals_15min_<YYYY>-<MM>.memmap`. Obs conversion tooling
   is still pending even at 1H (`rtma_ok_data_directories.md` s6).
3. **Nothing else.** Background files, norm factors, grid axes, and `elev_vars_ok.npy` are
   cadence-independent and carry over unchanged.

### Cost note

15-min is **4x the samples per unit time** vs. 1H (96 vs. 24 frames/day). One month goes from
~719 to ~2879 samples. The efficiency findings in `codex_context.txt` (2026-07-22) -- validation
every epoch, ViT attention over 8192 tokens at Grid B 384x192/patch 3, warm start that is not a
true resume -- bite proportionally harder. Address those before scaling cadence.

## Verification

- Pick a known timestamp; confirm `month_frame` maps it to the expected `(file, row)` and that
  obs and target rows correspond to the same time.
- `[DEBUG] era5_mode ... time_freq ... era5_frame_shape ...` (`loader.py:987-1000`) shows the
  expected channels.
- 6H/1D regression check: `get_time_aux` output unchanged (`minute == 0`).
- **Sample-count arithmetic** (worth knowing -- it is not a bug but surprises people):
  `dates = pd.date_range(start_date, end_date, freq=time_freq)` treats a bare `end_date` as
  `00:00`, so the final day contributes a single frame; `__len__` then returns
  `len(index) - 2` (`loader.py:887-888`). For `2022-01-01`..`2022-01-31` at `1H`:
  `721 - 2 = 719` samples -- matching the observed run. At `15min` the same range gives
  `2881 - 2 = 2879`. To include all of 31 Jan, pass `2022-02-01` as `end_date`.

## Known stale in-code comments (cosmetic, no functional effect)

These still say "15min" where the code now covers both sub-daily cadences:

- `loader.py:241-243` -- "On the rtma_surface 15-min path this slot is fed..."
- `loader.py:684` -- `_load_background_monthly` docstring, "the rtma_surface 15-min path"
- `loader.py:966` -- `NotImplementedError` message, "(rtma_surface + 15min)"
- `loader.py:1114-1115` -- "All 96 intraday samples of a day share that day's 00z field"
  (24 at `1H`)

## Related

- `plan_rtma_phase1.md` -- surface-only encoder (`--obs_set rtma_surface`), 5 obs -> 5 fields.
- `rtma_ok_data_directories.md` -- the concrete 1H data tree and its pending items.
- `note_climatology_normalization.md` -- why the background is normalized.
- `grid_config_ok.yaml` -- OK domain grid and per-month file templates.
