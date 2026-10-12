# SetConv length scales are declared learnable but are never trained

**Status:** static-analysis finding, independently reviewed and confirmed by a colleague
(2026-07-25); not yet confirmed at runtime (see "Verification"). No local Python on the dev box;
the one-line check below settles it in seconds on the cluster.

**Two corrections from that review are incorporated below:** (1) the five surface variables have
five *independent* frozen Parameters, not one shared Parameter; (2) `ascat_setconvs.init_ls` is
registered but its encoder bypasses it, so it does not learn either -- the same is true of
`sc_out.init_ls`.

## Summary

Every `convDeepSet` owns a Gaussian kernel length scale `init_ls`, created as an
`nn.Parameter` with `requires_grad = True`. The documentation and the code both present it as a
**learned** quantity.

In practice **7 of the 8 modality setconv groups are held in plain Python lists**, not
`nn.ModuleList`. PyTorch only registers submodules assigned directly as attributes or held in a
`ModuleList`/`ModuleDict`, so those parameters never enter `model.parameters()` -- and the
optimizer is built from exactly that. The length scales are therefore **frozen at their
initial value for the entire run**.

Nothing crashes and nothing is numerically wrong. `cmd_init_ls` simply behaves as a fixed
hyperparameter rather than a learned one, which is not what the code or `plan_obs.md` claims.

**For the RTMA-OK encoder this means all five surface length scales are fixed at whatever
`--cmd_init_ls` is passed.** The hand-tuned `2e-4` is load-bearing, not a starting point the
model refines.

## Evidence

| Layer | What it does | Reference |
|---|---|---|
| `convDeepSet.__init__` | `self.init_ls = nn.Parameter(...)`; `requires_grad = True` | `set_convs.py:21`, `:27` |
| `ConvCNPWeather.__init__` | `self.hadisd_setconvs = [convDeepSet(...) for _ in range(5)]` -- **plain list** | `models.py:147-150` |
| Same, all other modalities | amsua / amsub / hirs / sat / icoads / igra -- all plain lists | `models.py:130-158` |
| **Registered but unused** | `self.ascat_setconvs` and `self.sc_out` are single modules assigned directly, so both **are** registered -- but neither is ever called. `encoder_ascat` uses `nn.functional.interpolate` instead | `models.py:127`, `:160`, `:341` |
| Registration helpers | no `nn.ModuleList`, `add_module`, or `register_module` anywhere in the file | `models.py` (grep) |
| Optimizer | `torch.optim.Adam/AdamW(model.parameters(), ...)` | `trainer.py:55-65`, `:435-444` |
| Docs claim | "a single **learnable** length scale `init_ls` per setconv" | `plan_obs.md`, "Kernel + length-scale caveat" |

Net: **no length scale anywhere in the model is both registered and active.** The seven groups
that are actually used are unregistered; the two that are registered (`ascat_setconvs`, `sc_out`)
are dead code on the current forward path. Not one `init_ls` is trained.

## Consequences

1. **No optimizer update.** Autograd still computes a gradient and accumulates it into
   `init_ls.grad`; no optimizer step ever consumes it. The value never moves.
2. **Absent from `state_dict()`.** Checkpoints do not carry the length scales. Run-to-run
   consistency depends entirely on passing the same `--cmd_init_ls` flag. A warm start with a
   different flag value silently changes the kernel while appearing to "resume".
3. **Not moved by `model.to(device)`.** The parameter stays on CPU. This does not crash only
   because `compute_weights` does `self.init_ls.to(x1.device)` on every call
   (`set_convs.py:34`) -- a per-call host-to-device copy that also hides the problem.
4. **Not synchronized by DDP.** Harmless today (the value never changes, so ranks cannot
   diverge), but it would become a correctness issue the moment the parameter is made trainable
   without also registering it.

## Why it went unnoticed

The model trains perfectly well with a fixed kernel width -- a sensible constant length scale is
a reasonable inductive bias, and the global default (`0.001`, ~0.36 deg) was well matched to the
1.5 deg global grid. Nothing in the run log distinguishes "learned to 0.001" from "pinned at
0.001". The failure is silent by construction.

## Impact on RTMA-OK

`--cmd_init_ls 2e-4` (normalized) = `2e-4 * 360` = **0.072 deg** ~ 8 km, against a grid spacing of
0.027 deg lon / 0.0225 deg lat -- roughly **2.7 grid cells**. That is a good choice for a 2.5 km
surface analysis, and much better than the global default (0.36 deg ~ 16 cells, the
over-smoothing `plan_obs.md` warned about).

Note the mechanism precisely: the five variables have **five independent `nn.Parameter` objects**
(one per `convDeepSet`), not one shared parameter. They are simply all initialized to the same
`cmd_init_ls` and all frozen, so they can never diverge. Wind, temperature, and pressure have
materially different spatial decorrelation lengths, and a trained model would be expected to
separate them. The capacity to do so is present in the architecture; it is inert.

## Fix options

### Option A -- make them genuinely learnable

```python
self.hadisd_setconvs = nn.ModuleList([convDeepSet(...) for _ in range(N_HADISD_VARS)])
```
Apply to all seven plain-list groups. **This is not a free change:**

- Adds new entries to `state_dict`. **Existing warm-start checkpoints stay compatible**: the
  trainer already loads with `strict=False` (`trainer.py:97`, `:477`), so the new entries are
  simply absent from the checkpoint and start from the command-line value. (Caveat: `strict=False`
  also silently swallows genuinely missing weights -- worth logging the returned
  `missing_keys`/`unexpected_keys` rather than discarding them.)
- The length scales start moving mid-training, changing convergence behavior.
- **Weight decay would apply to `init_ls`** on the AdamW branch, which is wrong for a physical
  length scale (it would be pulled toward zero). Put them in a separate parameter group with
  `weight_decay=0.0`.
- Consider parameterizing `log(ls)` instead of `ls`: the kernel uses `ls**2`, so the sign is
  irrelevant, but an unconstrained `ls` can drift toward zero and degenerate into "each
  observation affects only its own cell."

### Option B -- make the freeze explicit (matches current behavior exactly)

Register `init_ls` as a **buffer** with `requires_grad=False`, and document it as a
hyperparameter. Zero behavior change; removes the misleading `requires_grad = True`, stops the
pointless gradient accumulation, and gets the value into `state_dict` so checkpoints become
self-describing. Update the `plan_obs.md` wording from "learnable" to "fixed hyperparameter".

### Agreed near-term approach

1. **Treat `cmd_init_ls` explicitly as a fixed, scientifically chosen hyperparameter** for the
   current RTMA experiments. Document it as such and stop describing it as learnable.
2. **Convert the SetConv lists to `nn.ModuleList`** regardless of the learn-vs-freeze decision,
   so the modules move with `model.to(device)` and appear in `state_dict`.
3. **If the scales stay fixed, register them as buffers** (`requires_grad=False`): they then
   travel with the model and are saved in checkpoints, making runs self-describing.
4. **If learned scales are wanted later:** positive parameterization (`log_length_scale`),
   exclude from weight decay, consider a smaller dedicated learning rate, and log each variable's
   scale per epoch so the behavior is observable.

Separately, decide what to do with `ascat_setconvs` (`models.py:127`) and `sc_out` (`:160`):
either wire them into the forward path or delete them. Registered-but-dead parameters are how
this class of confusion starts.

## Verification

One line, on any constructed model:

```python
print([n for n, _ in model.named_parameters() if "init_ls" in n])
```

- **Confirms the finding:** exactly two names -- `ascat_setconvs.init_ls` and `sc_out.init_ls` --
  both registered but never called on the current forward path. No `hadisd_setconvs` entry.
- **Refutes it:** one name per setconv across all modalities.

Stronger check -- train a few steps and compare:

```python
before = model.hadisd_setconvs[0].init_ls.item()
# ... N optimizer steps ...
after  = model.hadisd_setconvs[0].init_ls.item()
print(before, after, model.hadisd_setconvs[0].init_ls.grad)   # expect: equal, with a non-None grad
```

A non-`None` gradient alongside an unchanged value is the signature of this bug.

**Read that gradient with care.** `optimizer.zero_grad()` and `model.zero_grad()` both only clear
parameters they know about, and these are in neither. So `init_ls.grad` is a **running sum over
every backward pass since process start**, not a per-step gradient -- its magnitude says nothing
about how far the scale "wants" to move. This does not affect the fix (a fresh process builds
fresh Parameters with `grad=None`), but it will mislead anyone who inspects the value to estimate
step size.

## Related

- `plan_obs.md` -- the off-grid obs contract and the length-scale caveat (wording needs updating).
- `plan_hadisd_v_drop.md` -- same class of defect: internally consistent bookkeeping hiding
  silently discarded capability.
- `plan_rtma_phase1.md` -- where `--cmd_init_ls` is listed as an open decision for the OK domain.
