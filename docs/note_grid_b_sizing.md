# Grid B is oversized: 336 x 174 instead of 384 x 192

**Status:** static-analysis proposal, reviewed by a colleague (2026-07-25) and revised; not yet
run. The change is config-only (`aardvark/grid_config_ok_reduced_grid_B.yaml`); no code is
modified. Numbers below are derived from reading `models.py` and `vit.py` and should be confirmed
with one forward pass.

**Review outcome:** size arithmetic, token counts, patch-divisibility concern, and the
nearest-neighbour finding all **confirmed**. Four claims corrected and incorporated below:
(1) `strict=False` does **not** hide a pos-embed shape mismatch -- the warm start fails loudly;
(2) unsampled Grid-B cells still contribute indirectly via self-attention, so "never reach the
loss" was too strong; (3) the -37% applies to the quadratic attention terms only, not total
model FLOPs; (4) the fidelity improvement is a hypothesis, and the larger grid carries a real
latent-capacity advantage that the A/B must weigh.

## Proposal

Reduce the inner ViT grid (Grid B) for the OK domain from **384 x 192** to **336 x 174**.
Expected: **-21% ViT tokens** and **~-37% in the quadratic attention terms**. Total model FLOPs
and wall time will fall by *less* than 37%, since Q/K/V projections, MLP layers, normalization,
and patch decoding all scale linearly with token count -- though attention is large enough here
that the net saving should still be meaningful. Resampling geometry also improves substantially.
Two lines in a YAML; no code change.

## Background: the two grids

| | Grid A | Grid B |
|---|---|---|
| Size (OK) | 331 x 171 | 384 x 192 |
| Nature | **Geographic** -- real lon/lat degrees | **Dimensionless raster**; no coordinates |
| Origin | *Derived* from `urma_x_ok.npy` / `urma_y_ok.npy` | *Configured* in `grid_config_ok.yaml` |
| Carries | obs aggregation, background, elevation, target, model output | ViT activations only |

The encoder builds its input on Grid A, upsamples to Grid B, runs the ViT, then downsamples back
to Grid A (`models.py:516`, `:532`). Grid B exists for the transformer's convenience only --
nothing physical is defined on it.

## Constraints on Grid B (two enforced, one implicit)

1. **Even** -- `models.py:113-118` ("odd sizes break stride-2/pooling in the backbone").
2. **>= Grid A** -- `models.py:119-124` (oversample, never undersample).
3. **Divisible by `patch_size` (3)** -- **not asserted anywhere**, but required in practice:
   `vit.py:187-188` and `:236-237` compute the token grid with floor division
   (`img_size // patch_size`), so a non-divisible size silently **truncates** the trailing edge
   rather than padding it.

**Side finding on (3).** The global default Grid B is 256 x 128, which is *not* divisible by 3:
`256 = 85*3 + 1`, `128 = 42*3 + 2`. That configuration has been silently dropping one column and
two rows on every forward pass: `PatchEmbed` is a convolution with kernel = stride = patch size,
so incomplete trailing patches are simply omitted, and `unpatchify()` then reconstructs only
`floor(int_x/3)*3 x floor(int_y/3)*3` = 255 x 126.

**The same class of issue exists on the forecast path.** The forecast ViT uses `patch_size=5`
(`models.py:171`) at Grid A, and the common global grid is 240 x 121: 240/5 = 48 exactly, but
121/5 = 24.2, so one latitude row is dropped there too.

Both mean `plan_grids.md` s5 -- *"ViT pads for patch size, so exact divisibility is NOT
required"* -- is **incorrect for the current implementation** and should be revised. The OK
384 x 192 is clean (`128*3`, `64*3`), so current OK runs are unaffected.

## The arithmetic

Grid A is 331 x 171 = 56,601 cells. 336 = 112*3 and 174 = 58*3, so 336 x 174 is the **smallest
box satisfying all three constraints**.

| | 384 x 192 (current) | 336 x 174 (proposed) |
|---|---|---|
| >= Grid A / even / divisible by 3 | yes / yes / yes | yes / yes / yes |
| Grid B cells | 73,728 | 58,464 |
| ViT tokens | 128 x 64 = **8192** | 112 x 58 = **6496** (-21%) |
| Quadratic attention terms (~tokens^2) | 1.00x | **0.63x** (-37%) |
| Linear terms (QKV, MLP, norm, patch decode) | 1.00x | 0.79x (-21%) |
| Output cells not directly sampled by B->A | 17,127 (**23.2%**) | 1,863 (**3.2%**) |
| A->B resample ratio (lon) | 1.160x -- 53 duplicated columns | **1.015x -- 5 columns** |
| A->B resample ratio (lat) | 1.123x -- 21 duplicated rows | **1.018x -- 3 rows** |

## Two independent reasons this matters

### 1. Compute

Attention cost grows quadratically with token count, and `codex_context.txt` (2026-07-22) already
identifies the 8192-token global attention as the likely architectural hotspot. At 384 x 192,
**23% of the reconstructed Grid-B output cells are not directly sampled** by the
nearest-neighbour transfer back to Grid A; at 336 x 174 that falls to 3.2%.

State this precisely: those cells' *output values* do not appear in the final Grid-A field, but
their **latent tokens still participate in global self-attention** and can influence tokens that
are retained. So it is not pure waste -- it is compute whose contribution is indirect. The
earlier phrasing ("computed and thrown away, never reaching the loss") overstated the case and
has been corrected.

### 2. Resampling fidelity

Both transfers use `nn.functional.interpolate` with **no `mode` argument**, i.e. the default
`mode="nearest"` (`models.py:516`, `:532`; the same is true of the other three interpolate call
sites, `:341`, `:353`, `:536`). At a **non-integer 1.16x ratio**, nearest-neighbour resampling
duplicates and drops columns *irregularly*, with a repeating pattern of period ~6 cells. The final
analysis product is therefore a nearest decimation of a 384 x 192 field onto 331 x 171 -- a
plausible source of stair-stepping in exactly the sharp, terrain-driven gradients a 2.5 km
surface analysis exists to resolve. Moving to 1.015x cuts duplicated longitude positions from
~53 to ~5, which is geometrically far more natural.

**This is a hypothesis, not a demonstrated result.** Reduced duplication *should* reduce
artificial staircase patterns, but that must be confirmed by comparing output fields, not
assumed. There is also a genuine counter-argument: **the larger grid gives the transformer more
latent tokens and therefore more capacity.** Even where inputs are duplicated, the transformer
can build distinct latent representations in those cells. The smaller grid trades some latent
resolution for better geometric alignment and lower cost -- which is exactly why this needs to be
run as an A/B rather than adopted on arithmetic alone.

**Recommended regardless of the size change:** make `mode` explicit at all five interpolate call
sites. Even if `"nearest"` is retained, it should be a decision on the record rather than a
library default nobody selected.

## Risks and caveats

- **Positional embeddings do not transfer -- and the warm start will FAIL, loudly.** Changing
  Grid B changes the token grid (8192 -> 6496), so a 384 x 192 checkpoint carries
  `pos_embed: [1, 8192, 512]` where the new model expects `[1, 6496, 512]`.

  **Correction to an earlier draft of this note:** `strict=False` does **not** hide this.
  `strict` governs only *missing* and *unexpected* keys; a **shape mismatch on a key that exists
  in both** is appended to `error_msgs` inside `Module._load_from_state_dict`, and the
  `raise RuntimeError(error_msgs)` sits *outside* the `if strict:` block. So the load raises a
  size-mismatch error regardless. This is the desirable behavior -- the failure is loud.
- **`vit.interpolate_pos_embed` is not a drop-in fix for this checkpoint path** (`vit.py:57`).
  It expects a key named `net.pos_embed`, whereas the encoder checkpoint uses
  `decoder_lr.pos_embed`; it assumes a fixed 2:1 token-grid aspect ratio; and it may reverse the
  current Grid-B axis interpretation. For the first comparison, **fresh-start vs. fresh-start is
  the safe route.**
- **Not yet run.** The token arithmetic is inferred from `vit.py`'s floor divisions. One forward
  pass confirms it.

## How to test

`aardvark/grid_config_ok_reduced_grid_B.yaml` is a verbatim copy of `grid_config_ok.yaml` with
only `int_x`/`int_y` changed, so the comparison is clean. Swap one flag:

```bash
--grid_config ../aardvark/grid_config_ok_reduced_grid_B.yaml
```

Everything else (`--obs_set rtma_surface`, `--era5_mode rtma_ok_sfc`, `--in_channels 24`,
`--cmd_init_ls 2e-4`, `--time_freq 1H`) is unchanged.

**Controls:**

- **Fresh-start 384 x 192 vs. fresh-start 336 x 174.** Not a warm start -- the pos-embed shape
  mismatch makes that impossible anyway.
- Same **random seed, data period, batch size, learning rate, and epoch count**.
- **Do NOT change the interpolation mode during this A/B** -- that would confound the Grid-B
  comparison. Make `mode="nearest"` explicit as a *separate* cleanup commit.

**Measure:**

| Axis | Metric |
|---|---|
| Cost | wall time per batch and per epoch; peak GPU memory |
| Convergence | training loss vs. optimizer step (not vs. epoch) |
| Skill | RMSE and bias, per variable |
| Fidelity | fine-scale spatial structure, especially terrain-driven fields |

**Follow-up regardless of outcome:** add an explicit patch-divisibility assertion so future
configurations fail loudly instead of silently truncating (see the Global-grid finding above).

## Longer term: drop Grid B entirely

`plan_grids.md` s7 still lists "one grid or two?" as open, noting the forecast path already runs
the ViT directly at Grid A. Removing Grid B would eliminate both interpolations and the artifact
completely. Two obstacles:

- **331 is not divisible by 3** (331 = 110*3 + 1), so the ViT would truncate a column. The clean
  fix is to **pad** Grid A to 333 x 171 (= 111*3 x 57*3) and crop after, which is exact -- no
  resampling at all.
- **333 and 171 are odd**, so they fail the evenness guard at `models.py:113-118`. That guard is
  justified by "stride-2/pooling in the backbone", which is a UNet property; a plain ViT has no
  such requirement. Worth revisiting whether the constraint applies on the `vit_assimilation`
  path at all.

That path gives 6,327 tokens with zero resampling -- slightly better than 336 x 174 on both axes,
but it requires code changes, whereas 336 x 174 does not. Suggested order: take the config-only
win first, then evaluate the one-grid design.

## Related

- `plan_grids.md` -- Grid A derived vs. Grid B configured; s5 (padding claim, needs correction)
  and s7 (one-grid-vs-two, still open).
- `codex_context.txt` (2026-07-22) -- RTMA computational efficiency; identifies attention over
  8192 tokens as the likely hotspot.
- `aardvark/grid_config_ok_reduced_grid_B.yaml` -- the ready-to-run config.
