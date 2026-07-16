# HadISD meridional wind (`v`) is silently dropped by the encoder

## Summary

The encoder loads **5** HadISD surface variables but only encodes **4**. The 5th
variable — `v`, the meridional (north–south) wind component — is read from disk and even
given its own DeepSet layer, but is **never fed to the encoder**. Only `u` (zonal wind) is
used, so **half of the surface wind vector is silently discarded.**

This is *not* a channel-count error: the input-channel arithmetic is internally consistent
(the count function also assumes 4), so the model trains and runs without complaint. It is a
**data-loss / correctness bug**: information the pipeline went to the trouble of loading is
thrown away, almost certainly unintentionally.

## Evidence (three layers disagree on the variable count)

| Layer | Count it implies | Reference |
|---|---|---|
| Loader — variables loaded | **5** (`tas, tds, psl, u, v`) | `aardvark/loader.py:586` |
| Model init — DeepSet layers allocated | **5** (`N_HADISD_VARS = 5`) | `aardvark/models.py:74` |
| That same line's comment | "...vars **used by encoder_hadisd**" → 5 | `aardvark/models.py:74` |
| `encoder_hadisd` — variables actually encoded | **4** (`for channel in range(4)`) | `aardvark/models.py:198` |
| `expected_in_channels_assimilation` — HadISD channels | **4** (`hadisd = 2 * 4`) | `aardvark/train_module.py:91` |

Variable order is `["tas", "tds", "psl", "u", "v"]`, so `range(4)` encodes indices 0–3
(`tas, tds, psl, u`) and skips index 4 (`v`).

## Consequences

1. **Meridional wind `v` is discarded.** The encoder sees zonal wind only — an incomplete
   wind observation.
2. **`hadisd_setconvs[4]` is dead weight.** It is constructed but never called, so it
   receives no gradient and wastes a small amount of memory.
3. **The comment contradicts the code.** `N_HADISD_VARS = 5` is annotated "number of HadISD
   vars used by encoder_hadisd," but only 4 are used. This contradiction is the strongest
   signal that the `range(4)` is an oversight, not a deliberate "zonal-wind-only" choice.

## Why it has gone unnoticed

The channel bookkeeping is self-consistent: `expected_in_channels_assimilation` also hard-codes
`2 * 4 = 8` HadISD channels, matching the encoder's 8-channel output, so the published
`--in_channels` value (e.g. 241 for the global `4u_sfc` run) is correct *given* 4 variables.
Nothing asserts that the number encoded equals the number loaded — so the drop is invisible at
runtime.

## Fix plan

### Option A — include `v` (recommended if both wind components are wanted)

Three coupled edits (they must change together, or the channel count drifts):

1. `aardvark/models.py:198` — `for channel in range(4)` → `range(5)`
   (or, more robustly, `range(N_HADISD_VARS)` / `range(len(hadisd_vars))`).
2. `aardvark/train_module.py:91` — `hadisd = 2 * 4` → `2 * 5` (= 10).
3. `--in_channels` everywhere it is passed: **+2** per frame
   (global `4u_sfc`: 241 → 243; recompute for any other modality set; double the +2 if
   `--two_frames`).

### Option B — intentionally use zonal wind only (if that was the design)

Make the intent explicit and remove the waste:

1. `aardvark/loader.py:586` — drop `"v"` from `hadisd_vars` (stop loading it).
2. `aardvark/models.py:74` — `N_HADISD_VARS = 5` → `4`; fix the comment.
3. Leave the channel count at `2 * 4`.

### Preferred long-term: derive the count, don't hand-maintain it

Replace the three hard-coded `4`/`5`s with a **single source of truth** (the length of the
active HadISD variable list), and have `expected_in_channels_assimilation` and the encoder both
read from it. Add an assertion that *channels encoded == channels declared*. This makes the
class of bug (encode-count ≠ load-count ≠ declared-count) impossible and folds naturally into
the planned configurable-observations refactor.

## Verification after the fix

- Grep the run log for `[DEBUG] encodings[i] shape:` (printed once at
  `aardvark/models.py:401-404`) and confirm the HadISD block reports the expected channel count
  (10 for Option A, 8 for Option B).
- Confirm the concatenated input channel total equals the passed `--in_channels` (the model
  fails at the ViT input projection if they disagree).
- Sanity-check that `hadisd_setconvs` has no unused entries.

## Scope / ownership note

Decide Option A vs B based on **intent for surface wind**: a full analysis system (e.g. the
OK 2.5 km RTMA effort) will want both `u` and `v` → Option A. If the global model was meant to
ingest only zonal wind, Option B documents that. Either way the comment at `models.py:74`
must be reconciled with the code.
