# A-01 — Outer bicubic residual against direct prediction (D022)

Run 2026-08-13 on owner request. Ablation 3 of `docs/EXPERIMENT_PLAN.md`. Both architectures were
retrained from scratch with the additive bicubic path removed, on the frozen manifest, the frozen
seed, and the same 30,000-step schedule. **The frozen T-03 runs and `docs/RESULTS.md` are
untouched**; this is a separate document reporting a separate pair of runs.

The question is whether D006's outer form,

```
y_hat = bicubic(x) + R_theta(x)      against      y_hat = G_theta(x)
```

helps or hurts. D006 gave two reasons for it. One is a proof and is unaffected by anything here:
with the additive path, zeroed weights reproduce the bicubic baseline exactly, so any gain over
bicubic is attributable to the learned residual. The other — that it eases optimization — had never
been tested against a control. The specific hypothesis under test, from the owner, is that the
additive path imposes a bias the network then has to fight, which the `r`/`c` decomposition in
`docs/experiments/O-07-cross-resolution-transfer.md` made plausible by measuring correction magnitudes up to 9.5x too large at short
lead time.

## What was held fixed

| | value |
|---|---|
| Dataset | `swe_gaussian_32x128_v1`, processed manifest, IC registry `976e3a57…` |
| Split | 32 train / 8 validation / 8 test trajectories by seed ID |
| Schedule | `configs/experiment/full.yaml` — identical in every value to the frozen runs' `full_diagnostic.yaml`, verified programmatically; only the stage label differs |
| Seed | 20260812 |
| Steps | 30,000, cap reached by all four runs |
| Precision | BF16 (AVX512-BF16), AMD EPYC 9R14, 16 threads |
| Parameters | **identical between arms**: EDSR 1,517,571, U-Net 1,930,208 |

So exactly one factor varies. What "direct" means differs slightly by architecture, and that
matters when reading the result:

- **EDSR direct** has no interpolation anywhere. This is the published EDSR form; D006's outer
  residual had been added here only for parity with the U-Net.
- **U-Net direct** still bicubic-upsamples its *input*, because operating on the output grid is
  structural to the architecture. Only the additive skip is removed, so the tail must produce the
  absolute field. In image-SR terms this arm is SRCNN-style against the residual arm's VDSR-style.

## Runs

| | EDSR residual (frozen) | EDSR direct | U-Net residual (frozen) | U-Net direct |
|---|---|---|---|---|
| `run_id` | `20260812T230157Z_edsr_aae64836_077d6b53` | `20260813T191354Z_edsr_direct_c0b760ce_8dc7367c` | `20260812T235727Z_unet_e3ce47d7_da865691` | `20260813T191403Z_unet_direct_209183b2_8dc7367c` |
| Model config | `edsr_x4.yaml` | `edsr_direct_x4.yaml` | `unet_x4.yaml` | `unet_direct_x4.yaml` |
| Best epoch | 34 | 34 | 39 | 26 |
| Best validation | 0.079038 | 0.075436 | 0.039762 | 0.041125 |

Both new runs pass `scripts/verify_independent.py`: every field metric, the mass diagnostic, and
the aggregation protocol recomputed in plain numpy from the specification, all matching.

## Result 1 — for EDSR, direct prediction is better, and the interval excludes zero

Held-out test split, 1,576 snapshots over 8 trajectories, normalized macro-averaged MSE. The
comparison is **paired on trajectory**, which is essential here: the aggregate confidence
intervals span roughly ±0.03 while the arms differ by about 0.002, so overlapping intervals
could not have settled this either way.

| Arch | residual | direct | paired diff (direct − residual) | 95% CI | excludes 0 | direct wins |
|---|---:|---:|---:|---|---|---|
| EDSR | 0.0830 | **0.0813** | **−0.00166** | [−0.00315, −0.00033] | **yes** | 6 of 8 |
| U-Net | **0.0400** | 0.0421 | +0.00212 | [−0.00023, +0.00504] | no | 3 of 8 |

**EDSR: the owner's call was right.** Removing the additive bicubic path improves EDSR by a paired
2.0% relative, with an interval excluding zero and 6 of 8 trajectories agreeing. It is a small
effect, but it is real on this split and it is in the predicted direction.

**U-Net: no effect that this experiment can resolve.** The point estimate favours the residual arm
by 5.3% relative, but the interval includes zero and the sign test is 3 of 8 — the weakest possible
evidence. Read this as "no detectable difference", not as "residual is better".

The two arms' aggregate scores remain far better than either baseline (`nearest` 0.4301,
`bicubic` 0.4295), and both models' paired comparison against bicubic still excludes zero.

## Result 2 — the mechanism proposed for the gain is not the one that produced it

The hypothesis predicted the largest benefit at **short** lead time, where the needed correction is
smallest and an unconditionally added baseline should do the most damage. That is not what the
stratification shows.

| t (h) | bicubic | EDSR res | EDSR dir | U-Net res | U-Net dir |
|---:|---:|---:|---:|---:|---:|
| 2.01 | **0.0090** | 0.0560 | 0.0583 | 0.0103 | 0.0131 |
| 4.69 | 0.0402 | 0.0877 | 0.0818 | 0.0170 | 0.0205 |
| 7.37 | 0.0849 | 0.0762 | 0.0758 | 0.0212 | 0.0206 |
| 12.74 | 0.2155 | 0.0607 | 0.0550 | 0.0280 | 0.0305 |
| 15.42 | 0.2993 | 0.0545 | 0.0485 | 0.0319 | 0.0363 |
| 20.78 | 0.4453 | 0.0489 | 0.0447 | 0.0373 | 0.0378 |
| 23.46 | 0.5776 | 0.0510 | 0.0496 | 0.0431 | 0.0358 |
| 28.83 | 0.7748 | 0.0992 | 0.1010 | 0.0505 | 0.0531 |
| 34.19 | 0.9945 | 0.2143 | 0.2014 | 0.1170 | 0.1184 |

At the **shortest** lead time the direct arm is slightly *worse* for both architectures — EDSR
0.0583 against 0.0560, U-Net 0.0131 against 0.0103. EDSR's gain appears from about 4.7 h onward and
is largest in the 12–21 h band (0.0550 against 0.0607 at 12.74 h, 0.0485 against 0.0545 at 15.42 h).

So the direction of the aggregate effect is confirmed for EDSR while the proposed reason for it is
not: the additive path is not costing accuracy where the correction should be near zero. Whatever
the gain is, it is a mid-to-long-lead-time effect.

**And the short-lead-time deficit is not caused by the outer form.** At 2.01 h every one of the
four runs loses to bicubic's 0.0090: EDSR 6.2x (residual) and 6.5x (direct), U-Net 1.14x and 1.46x.
Removing the additive baseline does not fix it. That points the finger back at the objective —
equal-weight MSE averaged over a mixture of a well-posed and a partly unpredictable regime, with no
lead-time conditioning — rather than at the residual formulation. It also removes one candidate
explanation for the `docs/experiments/O-07-cross-resolution-transfer.md` over-correction finding.

The two architectures cross over at very different lead times, and the difference is large enough
that they should not be described together. **EDSR** is still behind bicubic at 4.69 h (2.0x, both
arms) and ahead by 7.37 h. **The U-Net is ahead by 4.69 h already** — 0.0170 and 0.0205 against
bicubic's 0.0402 — so its deficit region is confined to the first few hours. At the validation
split's full frame resolution the U-Net direct arm crosses bicubic at about **2.4 h**: 1.80x worse
at 2.01 h, 1.11x at 2.35 h, 0.94x at 2.51 h. An earlier revision of this document said both
architectures lose out to about 7 h, which was true only of EDSR.

Two reading traps in `runs/<run_id>/curves.png` are worth naming, because this is the figure the
deficit is easiest to miss in. Its third panel uses a **linear** y-axis, unlike the first two, and
that axis is scaled by bicubic's climb to about 1.1 — so at 2 h the model-versus-bicubic gap of
0.0064 occupies 0.6% of the panel height and is invisible. The panel also draws the **last** epoch
in bold, not the best one; `best.pt` here is epoch 26 while the bold curve is epoch 39. Those differ
in a way that matters: epoch 39 is better at every lead time below about 7 h (0.0145 against 0.0170
at 2.01 h) yet loses on the macro average that selects the checkpoint, because the average is
dominated by the long lead times where all errors are large. The selection rule inherits the
objective's regime-mixing problem.

## Result 3 — the physics diagnostic barely moves

Relative mass error, de-normalized SI (D014), bicubic 0.0392:

| Arch | residual | direct |
|---|---:|---:|
| EDSR | 0.0540 | 0.0527 |
| U-Net | 0.0347 | 0.0356 |

The outer form is not what makes EDSR degrade mass conservation relative to bicubic: the direct arm
degrades it too, 0.0527 against bicubic's 0.0392. This reproduces the headline finding of
`docs/RESULTS.md` — lower pixel error does not imply better physics — under a different
architecture-level choice, which strengthens the reading that it is a property of the MSE objective.

## What this does and does not license

- It does **not** supersede D006 or the frozen experiment. The frozen comparison stands as run.
- The comparison-fairness argument for the residual form is unaffected and remains a proof: the
  direct arms have no analogous property, and `tests/models/test_models.py` asserts that a zeroed
  direct model outputs zero rather than bicubic.
- One seed per arm. A 2.0% paired effect with an interval that just excludes zero
  (`[−0.00315, −0.00033]`) is the kind of result a second seed could soften. Treat "EDSR prefers
  direct prediction" as evidence on this split, not as an established property of the architecture.
- The four runs were not launched under identical machine load — the two ablation runs ran
  concurrently, alongside an unrelated job — so **no cost comparison is drawn here**. Accuracy is
  unaffected: training is seeded and deterministic given the config.
- Fresh workloads (`fresh_id`, `ring_ood`) and the 64→256 transfer test were **not** re-run for the
  direct arms. Whether direct prediction transfers better out of distribution is open, and it is a
  natural next step given that `docs/experiments/O-07-cross-resolution-transfer.md` found the over-correction largest there.

## Reproducing

```bash
for model in edsr unet; do
  cbrun -t rocky -- srun -c 16 -- python -m swe_sr.train \
      --config configs/model/${model}_direct_x4.yaml --experiment configs/experiment/full.yaml
done
python -m swe_sr.evaluate --run-dir runs/<edsr-direct> --split test
python -m swe_sr.evaluate --run-dir runs/<unet-direct> --split test
PYTHONPATH=$PWD python scripts/compare_ablation.py \
    --residual runs/<edsr-frozen> runs/<unet-frozen> \
    --direct   runs/<edsr-direct> runs/<unet-direct>
```

`scripts/compare_ablation.py` reads stored evaluation artifacts. It recomputes per-trajectory means
in memory for the two frozen runs, whose artifacts predate the field D021 added, and writes nothing
into them; the recomputation is checked against each artifact's own reported aggregate and the
script fails if they disagree. Both reproduced exactly: 0.082986 and 0.040033.

`PYTHONPATH=$PWD` is required on a machine with an editable install pointing at a different
checkout, since `python scripts/...` puts `scripts/` on the path rather than the repository root.
