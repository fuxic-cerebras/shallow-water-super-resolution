# A-03 — Does EDSR's "no normalization" advice transfer to ConvMixer?

Owner-requested, 2026-08-13. Decision record: D023.

## The question

Two papers this project builds on disagree on one point. EDSR's central claim is that batch
normalization **degrades** super-resolution, and `swe_sr/models/edsr.py` and `unet.py` both omit
normalization for that reason. ConvMixer measures BatchNorm as worth **1.44%** over LayerNorm on
CIFAR-10 and keeps it. D023 shipped the published ConvMixer block with BatchNorm intact and
flagged the contradiction rather than resolving it by assumption. This arm measures it.

## First result: BatchNorm cannot simply be deleted

The naive experiment — remove BatchNorm, change nothing else — is **untrainable**, and that is a
finding rather than an obstacle. ConvMixer's pointwise stage is `BN(GELU(PW(x)))`, with **no
residual**; only the depthwise stage has one (paper Eq. 2 / Eq. 3). So 16 unnormalized
non-residual layers compound, and gradient never reaches the early blocks. EDSR does not have
this problem because *every* EDSR block is residual, which is why it trains unnormalized at
depth 16.

First-block depthwise gradient norm, relative to the last block, measured at initialization:

| variant | first/last gradient | activation std | trainable |
|---|---:|---:|---|
| BatchNorm, as published | 13 | 1.000 | yes |
| normalization removed, nothing else | 3.9e-07 | 0.021 | no — gradient dies |
| + residual around the pointwise | 0.81 | 49.5 | no — activations diverge |
| + residual and `res_scale = 0.1` | 0.90 | 0.409 | **yes** |

So the arm changes three values together, which is exactly EDSR's recipe — residual in every
block, residual scaling, no normalization:

```yaml
normalization: none        # batch  -> none
pointwise_residual: true   # false  -> true
res_scale: 0.1             # 1.0    -> 0.1
```

**Read this as "ConvMixer as EDSR would have designed it" against "ConvMixer as published",
not as a single-factor BatchNorm ablation.** It cannot be the latter. Everything else — width,
depth, kernel, patch size, decoder, manifest, seed, schedule — is identical.
`test_removing_normalization_alone_would_not_train` pins the measurement so that a future
simplification to a "clean" one-factor ablation fails loudly instead of producing a dead run.

Capacity differs by 0.98% (1,703,171 against 1,720,067). That is unavoidable: removing
BatchNorm removes its affine parameters. The residual and the scaling add none.

## Result: BatchNorm wins, decisively

Held-out test split, normalized macro-averaged MSE. Both runs pass
`scripts/verify_independent.py` (metrics recomputed in plain numpy: PASS).

| arm | test MSE | 95% CI | params | best epoch | epochs run |
|---|---:|---|---:|---:|---:|
| ConvMixer, BatchNorm as published | **0.0651** | [0.0366, 0.0963] | 1,720,067 | 21 | 36, early-stopped |
| ConvMixer, EDSR-style unnormalized | 0.1120 | [0.0772, 0.1469] | 1,703,171 | 36 | 39, full budget |

Paired on trajectory, the protocol `docs/VALIDATION.md` prescribes (positive favours BatchNorm):

| paired diff | 95% CI | excludes 0 | BatchNorm wins |
|---:|---|---|---|
| +0.04692 | [+0.01308, +0.08075] | yes | 7 of 8 trajectories |

BatchNorm is **1.72× better** and the interval excludes zero.

## Reading it

The direction matters for how much weight this carries. Because the unnormalized arm bundles
three changes, a win for *it* would have been ambiguous — impossible to attribute to
normalization rather than to the residual or the scaling. BatchNorm won instead, and it won
against a recipe specifically constructed to work without normalization. That is the
unambiguous direction, and it is a genuine negative result for transferring EDSR's guidance to
this architecture.

The failure mode is **underfitting, not overfitting**. The unnormalized arm's final training
loss is 0.1082 against the BatchNorm arm's 0.0103 — an order of magnitude worse on data it was
free to memorize — and its validation loss flatlined at 0.1181 from epoch 36 to 39. It used its
entire 30,000-step budget without early stopping and still never fit. `res_scale = 0.1` on both
residual paths is the likely cause: it is what makes the unnormalized network trainable at all,
and it also damps every block's contribution. BatchNorm's advantage here is optimization, not
regularization, which is consistent with EDSR's objection to BatchNorm (that it discards range
flexibility) simply not biting in an architecture whose non-residual stage depends on rescaling.

Two honest caveats. `res_scale` was chosen as EDSR's published 0.1 with no tuning, so some of
the 1.72× gap may be that value rather than normalization per se; a sweep is the obvious
follow-up (A-06). And the arm is not parameter-matched, though 0.98% is far too small to
explain the gap.

## Why EDSR survives the same removal and ConvMixer does not

"It underfits" is the symptom, not the reason, and the reason matters: if EDSR trains fine
unnormalized at the same depth on the same data, something specific to ConvMixer must be doing
the damage. Two things are, and both are structural.

**1. ConvMixer's residual branch ends in a nonlinearity; EDSR's ends in a linear convolution.**
GELU output is mostly positive, so `x + GELU(DW(x))` injects a positive-mean term at every one
of this arm's 32 residual adds, and a DC offset accumulates with nothing to remove it. EDSR's
`x + Conv(ReLU(Conv(x)))` terminates on a convolution whose weights are symmetric about zero,
so the branch is zero-mean and adding it 16 times drifts nowhere. In the *published* ConvMixer
block BatchNorm sits immediately after that GELU, so its mean subtraction is precisely the
corrective; removing it leaves the network with no mean-centering anywhere.

Activation mean through the trained bodies, on identical input (`|mean|/std` in brackets):

| after | ConvMixer BatchNorm | ConvMixer no norm | EDSR no norm |
|---|---:|---:|---:|
| stem / head | +0.039 (0.04) | +0.152 (0.39) | -0.020 (0.03) |
| block 4 | +0.216 (0.16) | +0.197 (0.42) | -0.019 (0.03) |
| block 8 | -0.086 (0.10) | +0.245 (0.41) | -0.015 (0.02) |
| block 12 | -0.114 (0.14) | +0.322 (0.37) | -0.033 (0.04) |
| block 16 | +0.042 (0.04) | **+0.402 (0.41)** | -0.015 (0.02) |

The unnormalized ConvMixer drifts monotonically to a DC offset worth 40% of its own signal
amplitude. EDSR is flat at 2-4%. The drift is architectural, not learned: at initialization the
same 16-block stack accumulates +0.365 with the branch ending in GELU and +0.002 with the
branch ending in a linear convolution, about +0.023 per block either way of that difference.

**2. Depthwise convolution never mixes channels, so per-channel scale drift is uncorrected.**
A depthwise kernel scales each channel by its own private weights, and nothing re-equalizes
them. EDSR's dense 3x3 sums over all 64 input channels at every layer, which averages channel
scales as a side effect. Measured ratio of largest to smallest per-channel variance in the
trained bodies: **500-4300x** for the unnormalized ConvMixer against **2-4x** for EDSR.
BatchNorm is per-channel, so once again it is exactly the missing corrective.

Both properties follow from ConvMixer being depthwise-separable and post-activation. **EDSR's
finding is real but architecture-conditional:** "batch normalization degrades
super-resolution" holds for dense residual networks whose branches end linear, and does not
transfer to a design where BatchNorm is carrying the scale and mean control.

One task-specific aggravator is worth naming. This model predicts a residual over bicubic on
zero-mean normalized channels, so the target correction is itself ~zero-mean. A DC offset at
40% of signal amplitude must therefore be cancelled downstream by the decoder, which spends
capacity undoing an artifact instead of reconstructing detail. That is consistent with the
observed failure being underfitting (final train loss 0.1082 against 0.0103) rather than
overfitting, and it is why A-07 is the more interesting follow-up than A-06.

## What the arm gains

One real property: with no batch statistics anywhere, the unnormalized arm is batch-independent
in **training** mode too, not only under `eval()`. The published arm couples samples while
training, which is why every reporting path calls `model.eval()` (D023). That cost is
now measured as buying a 1.72× accuracy improvement, so it is worth paying — but the tradeoff is
real and is why the eval-mode contract is asserted in tests rather than assumed.

## Reproducing

```bash
cbrun -t rocky -- sbatch scripts/slurm_convmixer.sbatch convmixer_nonorm full
python -m swe_sr.evaluate --run-dir runs/<run> --split test
PYTHONPATH=$PWD python scripts/verify_independent.py --run-dir runs/<run>
```

`PYTHONPATH` is required for `scripts/*.py` in this environment: an editable install resolves
`swe_sr` to a sibling checkout, so `python scripts/foo.py` can silently import different code.
`python -m` is unaffected because the working directory takes precedence.

Runs: `20260813T232258Z_convmixer_473b97bc_116ef1a8` (published),
`20260814T001237Z_convmixer_nonorm_d8cbb386_116ef1a8` (unnormalized). Frozen T-03 artifacts and
`docs/RESULTS.md` headline rows are untouched.
