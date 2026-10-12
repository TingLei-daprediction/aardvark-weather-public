# Plan: Ingesting Surface Observations (Off-Grid) for the OK 2.5 km Domain

How Aardvark handles observation *locations*, and the contract for adding Oklahoma surface
observations to the encoder. Scope: the **encoder / assimilation** path.

## 1. How the current model handles observation locations

The encoder reads each sample as (location, value) pairs and aggregates them onto the
data/target grid (Grid A) via DeepSet "setconv" layers. Two coordinate regimes exist today
(see `loader.py:get_index`, ~L829-899):

- **Off-grid point obs** (`OffToOn` setconv):
  - **ICOADS** (ships/buoys): *time-varying* locations -- both `icoads_x[index]` and
    `icoads_y[index]` are indexed per sample (`loader.py:838-844`). A flag
    `icoads_x_is_static` can force static coords.
  - **HadISD** (surface stations) and **IGRA** (radiosondes): *static* locations
    (`self.hadisd_x` at `loader.py:893`; `self.igra_x` at `loader.py:876`) with per-sample
    values; missing obs handled by NaN on the values.
- **Pre-gridded obs** (`OnToOn` setconv): AMSU-A/B, HIRS, IASI, ASCAT, GRIDSAT are regridded
  to a fixed grid before loading (e.g. `AMSUA_Y_SHAPE_1D=(4746,121,240,11)`), so their
  "locations" are fixed grid cells; per-sample variation is which cells are valid.

**Conclusion:** the architecture fully supports sample-dependent (time-changing) observation
locations -- ICOADS proves the end-to-end path. Fixed-network obs simply reuse static coords.
For OK surface obs we use the `OffToOn` path: HadISD pattern (static `x`) if the station
network is fixed, or ICOADS pattern (per-sample `x`) if station membership/positions vary.

## 2. The OffToOn contract (from `set_convs.py`)

`convDeepSet.forward(x_in, wt, x_out)` with `mode="OffToOn"`:

| Arg     | Meaning                                   | Shape                       |
|---------|-------------------------------------------|-----------------------------|
| `x_in`  | observation coordinates `[lon, lat]`      | each `(batch, N)`           |
| `wt`    | observation values (one setconv/channel)  | `(batch, 1, N)`             |
| `x_out` | target grid `self.int_grid` `[lon, lat]`  | `(1, nlon)`, `(1, nlat)`    |

`N` = padded number of obs points per sample. One `convDeepSet` per variable/channel.

### Coordinate convention (critical)
The kernel computes a raw Euclidean distance between `x_in` and `x_out`
(`set_convs.py:30-42`), so observation coordinates **must be in the same normalized units as
`int_grid`**: degrees / 360 (loader divides by `LATLON_SCALE_FACTOR`). After the grid fix,
`int_grid = era5_x/360, era5_y/360` (lon in [0, ~0.996], lat in [-0.25, 0.25]).
- Feed normalized coords, never raw degrees.
- Longitude must be on the **0-360 branch** matching `era5_x` (no negatives). OK is
  ~ -94..-103 E -> convert to ~257..266 before dividing by 360.

### NaN semantics (missing data AND padding)
1. **Missing value** -> NaN in `wt`. The density channel is set to 0 where `wt` is NaN
   (`set_convs.py:47-48`) and the NaN value is zeroed (`:51`): the obs contributes nothing.
2. **Padded/absent point** -> NaN in `x_in`. Lon/lat NaN masks zero those points' weights
   (`set_convs.py:56-65`). This is the padding mechanism: pad to fixed `N`, fill unused
   slots with NaN coords (and NaN values).

### Batching
`N` must be uniform across the batch -> pad to a fixed max point count (ICOADS uses 12000).
Choose a max covering the densest OK cycle.

### Kernel + length-scale caveat
Weight = separable Gaussian RBF `exp(-0.5 * dist^2 / ls^2)` with a single **learnable length
scale `init_ls` per setconv** (`set_convs.py:34`), applied to lon/lat separately then
combined (`:67`). Default `cmd_init_ls=0.001` (normalized) ~= **0.36 deg** -- tuned for the
1.5 deg global grid. On a 2.5 km (~0.0225 deg) OK grid that is ~16 cells wide and will
over-smooth surface obs initially. **Initialize a much smaller `--cmd_init_ls`** for the
regional domain (already exposed as the `--cmd_init_ls` CLI arg ->
`ConvCNPWeather(cmd_init_ls=...)`).

### Output format
Each setconv returns **density + density-normalized value** per channel
(`set_convs.py:89-100`). So a modality with `C` variables contributes **2*C** channels to the
encoder input (the "channel doubling").

## 3. Recipe to add an OK surface-obs modality

1. **Loader**: build per-sample (or static) `obs_x = [lon, lat]` normalized `/360` (lon on
   0-360 branch) and `obs_y` of shape `(C, N)`, padded to a fixed `N` with NaN for both
   coords and values. Add them to the task dict, mirroring `icoads_x`/`icoads_y` (per-sample)
   or HadISD (static).
2. **Model** (`models.py`): add
   `self.oksfc_setconvs = [convDeepSet(cmd_init_ls, "OffToOn", density_channel=True,
   device=...) for _ in range(C)]`, an `encoder_oksfc()` modeled on `encoder_icoads`
   (`models.py:225-241`), and include it in the `encodings` list in `forward`.
3. **Channel counts**: add `2*C` to `expected_in_channels_assimilation`
   (`train_module.py:90`) and to the encoder `--in_channels`. (See `plan_static.md` for the
   "source of truth" note on these functions.)
4. **Length scale**: set a small `--cmd_init_ls` appropriate to 2.5 km.

## 4. Decisions for colleagues

- **Static vs per-sample locations** for OK surface obs: fixed mesonet/METAR network ->
  HadISD pattern (static `x`, simplest). Variable membership / moving platforms -> ICOADS
  pattern (per-sample `x`, `icoads_x_is_static=False`).
- **Padded max point count `N`**: size to the densest cycle; trades memory for headroom.
- **Per-variable vs shared station sets**: HadISD uses a separate station array per variable
  (`models.py:198-199`); decide whether OK variables share one station list or not.

## 5. Performance note

The einsum is `O(N * nlon * nlat)` per channel. A fine OK grid times a large padded `N` can
be memory-heavy -- size `N` and the grid deliberately and check peak memory on one batch
before scaling.

## Related

- `plan_static.md` -- static (elevation) inputs; channel-count "source of truth".
- `plan_grids.md` -- configurable grids; `int_grid` now derived from `era5_x/era5_y`.
