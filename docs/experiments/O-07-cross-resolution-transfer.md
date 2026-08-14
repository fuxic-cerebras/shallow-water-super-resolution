# O-07 — Cross-resolution transfer: 32→128 models tested on 64→256

Run 2026-08-13 on owner request (O-07). The two frozen T-03 checkpoints were evaluated, without
any retraining or tuning, on a **newly generated 64→256 paired dataset**. Nothing was trained on
64→256 at any point.

This is a scale-transfer test, not an ordinary evaluation. The upscaling factor is the same ×4,
the domain is the same 1,000 km basin and the initial-condition registry is byte-identical
(`976e3a57…` on both pairs), but the absolute grid spacing halves: coarse `dx` goes 32.26 km →
15.87 km, so the same physical bump spans about 2.0 coarse cells on the trained pair and 4.1 on
the transfer pair. A convolutional model has a fixed receptive field in *pixels*, so the
question is whether the learned operator survives that change.

## What is and is not controlled

| | 32→128 | 64→256 |
|---|---|---|
| IC registry hash | `976e3a57…` | `976e3a57…` (identical) |
| Splits | 32/8/8 by seed ID | identical membership |
| Snapshots per trajectory | 197 | 197 |
| Lead-time window | 2.0 – 34.86 h | 2.0 – 34.72 h |
| Shared time step | 25.139798 s | 12.520605 s |
| Coarse `dx` | 32.26 km | 15.87 km |
| σ/`dx` at σ = 65 km | ≈ 2.0 | ≈ 4.1 |

**Normalization is not a confound, and this was measured rather than assumed.** Each pair carries
its own train-split statistics (D019), and `check_pair_id` refuses to apply one pair's statistics
to another. Those statistics turn out to be nearly identical — `eta` std 0.17616970 against
0.17701276 (0.48% apart), `u` 0.03945340 against 0.03962224 (0.43%) — so re-standardizing changes
amplitude by well under 1% and the comparison isolates **spatial scale**. A second variant using
the frozen 32→128 statistics verbatim was therefore not run; at a 0.5% scale difference it could
not separate from this one.

The 64→256 release passes all 12 data gates (`python -m swe_sr.data.validate`), worst relative
mass drift 6.03e-15 against a 1e-12 tolerance. Its manifest records commit `8265349`, which is
HEAD at manifest-write rather than at launch; `git diff 400066e 8265349 -- swe_sr/solver/
swe_sr/data/ configs/data/` is empty, so the generation path was byte-identical throughout.

## Result 1 — both models still beat bicubic, by much less

Test split, 1,576 snapshots over 8 trajectories, normalized macro-averaged MSE, 95% trajectory
bootstrap:

| Method | 32→128 (trained) | 64→256 (transfer) |
|---|---:|---:|
| nearest | 0.4301 | 0.1076 |
| bicubic | 0.4295 | 0.1070 |
| EDSR | 0.0830 `[0.0543, 0.1129]` | **0.0444** `[0.0310, 0.0582]` |
| U-Net | **0.0400** `[0.0261, 0.0544]` | 0.0497 `[0.0350, 0.0649]` |
| EDSR skill vs bicubic | 5.2× | 2.4× |
| U-Net skill vs bicubic | 10.7× | 2.2× |

Paired bootstrap against bicubic on the transfer pair, both excluding zero:
EDSR `-0.0626 [-0.0790, -0.0443]`, U-Net `-0.0572 [-0.0722, -0.0400]`.

**Read the absolute numbers with care.** Both models score *lower* on 64→256 than on the pair
they were trained on, which looks like transfer improving things. It is not: the transfer task is
intrinsically about four times easier, because bicubic also drops from 0.4295 to 0.1070. At
σ/`dx` ≈ 4.1 the coarse solve already resolves the structure that at σ/`dx` ≈ 2.0 it destroys.
The skill ratio against bicubic on the same data is the quantity that transfers, and it falls.

## Result 2 — the ranking inverts, and this one is paired-tested

In distribution U-Net beats EDSR by 2.1×. On 64→256 **EDSR wins**: 0.0444 against 0.0497. The two
CIs overlap heavily, so overlapping intervals cannot settle it, and the models are evaluated in
separate processes against separate run directories. The per-trajectory means are now serialized
(`trajectory_means_macro_mse_normalized`) specifically so this comparison can be paired at all:

```
paired bootstrap, unet - edsr on 64->256:  +0.00533  95% CI [+0.00268, +0.00826]   excludes 0
sign test:                                 U-Net worse on 8 of 8 trajectories
```

So the inversion is real, not an artifact of interval overlap. It is also the **second**
independent instance of the same pattern: on the unseen annular `ring_ood` family EDSR likewise
beat U-Net (0.3246 against 0.3505). U-Net's in-distribution advantage does not survive either an
initial-condition shift or a resolution shift. A plausible reading, consistent with
`docs/ARCHITECTURE.md`'s expected-strength table, is that U-Net's multi-scale encoder ties its
learned features to absolute pixel scale more tightly than EDSR's 16 local residual blocks do —
but that is an interpretation of two data points, not something this experiment establishes.

## Result 3 — both models lose to bicubic at short lead time, on *both* pairs

This is the finding the aggregate hides, and the stratified table is the honest reading:

| t (h) | bicubic 128 | EDSR 128 | U-Net 128 | bicubic 256 | EDSR 256 | U-Net 256 |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.0090 | 0.0560 | 0.0103 | **0.0022** | 0.1134 | 0.1599 |
| 7.3 | 0.0849 | 0.0762 | 0.0212 | 0.0179 | 0.0472 | 0.0682 |
| 12.7 | 0.2155 | 0.0607 | 0.0280 | 0.0451 | 0.0349 | 0.0445 |
| 23.4 | 0.5776 | 0.0510 | 0.0431 | 0.1366 | 0.0282 | 0.0249 |
| 34.1 | 0.9759 | 0.2109 | 0.1173 | 0.2789 | 0.0844 | 0.0537 |

At 2 h on the transfer pair bicubic scores 0.0022 while EDSR scores 0.1134 — **52× worse** — and
U-Net 0.1599, **73× worse**.

In distribution the same deficit exists but is far milder and differs sharply by model: EDSR loses
to bicubic out to about 7 h (0.0560 against 0.0090 at 2 h, 6.2×), while U-Net is only marginally
behind at 2 h (0.0103 against 0.0090, 1.14×) and ahead by 7 h. Transfer pushes **both** models'
break-even out to about 12–13 h and makes the early deficit an order of magnitude worse. So the
short-lead-time deficit is not created by transfer; it exists in distribution, and transfer
amplifies it and equalizes it across the two architectures.

## Result 4 — the reason is correction *magnitude*, not direction

`scripts/analyze_transfer.py` decomposes this with an identity rather than an estimate. Writing
the correction bicubic needs and the one the model applied,

```
need = y - bicubic(x)        applied = y_hat - bicubic(x)
r = ||applied|| / ||need||   c = cos(applied, need)
```

expanding `||y_hat - y||²` gives **exactly**

```
MSE_model / MSE_bicubic = r² - 2rc + 1
```

which is minimized at `r = c`, where it equals `1 - c²`. So `c` bounds what the correction could
achieve if only its magnitude were fixed, and the gap between the measured ratio and `1 - c²` is
pure magnitude miscalibration. The identity is checked per snapshot, not assumed: worst deviation
3.9e-08 across all four model-pair combinations, 400 snapshots each, which is float error alone.

| | t = 2 h | | | t ≈ 23 h | | |
|---|---:|---:|---:|---:|---:|---:|
| | r | c | ratio | r | c | ratio |
| EDSR 32→128 | 2.97 | 0.559 | 7.22 | 0.91 | 0.959 | 0.085 |
| EDSR 64→256 | **7.97** | 0.357 | **61.5** | 0.91 | 0.888 | 0.215 |
| U-Net 32→128 | 1.56 | 0.759 | 1.30 | 0.97 | 0.967 | 0.074 |
| U-Net 64→256 | **9.51** | 0.385 | **87.2** | 1.09 | 0.914 | 0.198 |

Two things follow.

**The catastrophic short-lead-time loss is over-correction, and it is recoverable in principle.**
At 2 h on the transfer pair the models apply corrections 8–9.5× too large. Yet `1 - c²` is
0.868 (EDSR) and 0.845 (U-Net) — *below 1* — so a single scalar per lead time would turn an 87×
loss into a modest win. Nothing about the direction of the correction is broken; the model has
simply learned a correction sized for the error scale it trained on, and applies it to an input
that barely needs one. At long lead time `r ≈ c` and the models are near-optimally calibrated.

**The genuine, unrecoverable part of the transfer loss is the alignment drop.** At t ≈ 23 h, `c`
falls from 0.959 to 0.888 (EDSR) and from 0.967 to 0.914 (U-Net), so the floor `1 - c²` rises from
0.0794 to 0.2120 and from 0.0645 to 0.1639 — 2.7× and 2.5×. Those factors are close to the
observed aggregate skill losses (5.2× → 2.4×, a factor of 2.1; and 10.7× → 2.2×, a factor of 4.9),
so alignment loss accounts for most of EDSR's degradation. It does **not** fully account for
U-Net's, whose aggregate skill falls further than its alignment alone predicts — the remainder sits
in the short-lead-time over-correction, which is far worse for U-Net (r = 9.51 against EDSR's
7.97 at 2 h) and which the aggregate mixes in. **No rescaling can recover the alignment part.**

**This is a hypothesis about a fix, not a fix.** A per-lead-time rescaling would need lead time
at inference, which the model is not given and which a deployed system may not know; and it was
not tried, so it is untested. It is stated because the decomposition makes it a concrete,
falsifiable next experiment, not because it is a result.

## Result 5 — physical consistency degrades again

Relative mass error, de-normalized SI (D014): on 64→256 bicubic 0.0197, EDSR 0.0471, U-Net 0.0439.
Both models make mass conservation **worse** than the baseline they beat on pixel MSE, by a factor
of about 2.3. This reproduces the in-distribution finding on a dataset neither model has seen, so
it is a property of this training objective rather than a quirk of one split: equal-weight
normalized channel MSE (D005) does not constrain the physical diagnostic, and optimizing it
degrades mass conservation.

## Figures

Not committed — `viz/` is gitignored. Regenerate:

```bash
python scripts/plot_transfer.py --runs runs/<edsr-run> runs/<unet-run>
```

| File | Contents |
|---|---|
| `viz/transfer_leadtime.png` | normalized macro MSE against lead time, one panel per pair on shared axes; the bicubic crossings are the result |
| `viz/transfer_decomposition.png` | `r` against `c`, and measured MSE ratio against the `1 - c²` floor; the 1.0 line is bicubic, so every crossing is a change of verdict |
| `viz/transfer_qualitative.png` | `eta` fields at four lead times at 256², coarse input through both models to target |
| `viz/transfer_qualitative_error.png` | signed errors for the same frames, each panel labelled with its MSE |

## Reproducing

```bash
python -m swe_sr.data.generate --config configs/data/backup_64x256.yaml
python -m swe_sr.data.validate --manifest data/processed/swe_gaussian_64x256_v1/manifest.json

for run in runs/<edsr-run> runs/<unet-run>; do
  python -m swe_sr.evaluate --run-dir $run --split test \
      --manifest data/processed/swe_gaussian_64x256_v1/manifest.json
  python scripts/analyze_transfer.py --run-dir $run --stride 4 \
      --manifest data/processed/swe_gaussian_64x256_v1/manifest.json \
      --out $run/decomposition_test__swe_gaussian_64x256_v1.json
done
```

A cross-pair evaluation writes `evaluation_test__<pair_id>.json`, never the canonical
`evaluation_test.json`. That naming is derived from the data rather than passed as a flag, because
the earlier behaviour would have overwritten the frozen T-03 result with transfer numbers; see the
commit that introduced the guard and its negative test.

## Limits of this experiment

- **One transfer direction, one factor.** 32→128 to 64→256, ×4 throughout. Nothing here says what
  happens at ×2 or ×8, or transferring downward to a coarser pair.
- **The decomposition is subsampled**: every 4th frame, 400 of 1,576 snapshots, all 8 trajectories
  and the full lead-time range. The aggregate tables use all 1,576.
- **The two pairs' saved times differ by 0.392%** (600.989 s against 603.355 s per interval,
  because the spacing ratio is 127/255 rather than exactly 1/2). Lead-time rows are therefore
  aligned to within about 0.4%, not exactly. Recorded in the Open Questions of
  `docs/PROJECT_SPEC.md` since before this experiment.
- **No retraining was attempted.** Whether a model trained on 64→256, or on both pairs, closes the
  gap is exactly O-01, still deferred.
- **`ms/frame` is not compared across pairs here** — it is host-load dependent, as the correction
  in `docs/SIGNOFF.md` records.
- Single-agent verification, with the same limitation `docs/SIGNOFF.md` states: the decomposition
  identity is self-checking per snapshot, but no independent reviewer has read this analysis.
