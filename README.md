# shallow-water-super-resolution

Neural spatial super-resolution for a 2D shallow-water solver. A coarse numerical solution is
the model input; an **independently integrated** fine-grid solution at the same physical time is
the target:

```
F_theta: R^(3 x 32 x 32)  ->  R^(3 x 128 x 128)
```

Channels are `[eta, u, v]` — surface elevation and depth-averaged velocity. Total water height
is `h = H + eta`; `eta` is never renamed to `h`.

The coarse input is a genuine coarse PDE solve, not a downsampled fine field (D002). That is the
point of the study, and it has a consequence worth knowing before reading any number: the two
solves progressively decorrelate, so baseline error grows strongly with lead time and every
result is reported stratified by it.

## Results

The frozen primary experiment is `docs/EXPERIMENT_FREEZE.md` (T-03, D020); generated results are
in `docs/RESULTS.md`. Held-out test split, 1,576 snapshots over 8 trajectories, normalized
macro-averaged MSE with a 95% trajectory bootstrap interval:

| Method | Params | normMSE | 95% CI |
|---|---:|---:|---|
| nearest | 0 | 0.4301 | [0.3076, 0.5435] |
| bicubic | 0 | 0.4295 | [0.3069, 0.5431] |
| EDSR x4 | 1,517,571 | 0.0830 | [0.0543, 0.1129] |
| U-Net x4 | 1,930,208 | **0.0400** | [0.0261, 0.0544] |

Normalized channels have unit variance, so predicting the channel mean scores exactly 1.0.

Three findings beyond the headline:

- **U-Net beats EDSR by 2.1x** in distribution, for 1.27x the parameters and 2.2x the inference
  cost. A 2,000-step pilot had them tied, so this only emerged at the full 30,000 steps.
- **Lower pixel error does not imply better physics.** U-Net improves relative mass error over
  bicubic (0.0347 against 0.0392) while EDSR makes it *worse* (0.0540) despite beating bicubic
  fivefold on MSE.
- **The ranking inverts out of distribution.** On the unseen annular `ring_ood` family, EDSR
  scores 0.3246 against U-Net's 0.3505, so U-Net's in-distribution advantage does not transfer.

## Getting started

```bash
git clone --recurse-submodules <this repo>        # the solver reference is a pinned submodule
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,viz]"
./scripts/check.sh                                # lint, format, types, fast tests
```

`scripts/check.sh` is the single gate; CI runs exactly it, so local and CI cannot drift.

## Commands

```bash
python -m swe_sr.data.generate  --config configs/data/primary_32x128.yaml
python -m swe_sr.data.validate  --manifest data/processed/swe_gaussian_32x128_v1/manifest.json
python -m swe_sr.train          --config configs/model/unet_x4.yaml \
                                --experiment configs/experiment/full.yaml
python -m swe_sr.evaluate       --run-dir runs/<run-id>
python -m swe_sr.evaluate_fresh --run-dir runs/<run-id> --scenario ring_ood
python -m swe_sr.report         --runs runs/<edsr-run> runs/<unet-run> --out docs/RESULTS.md

python scripts/verify_independent.py --run-dir runs/<run-id>  # recomputes metrics from arrays
python scripts/visualize.py --seed 0 --all                    # animations via reference viz_tools
python scripts/plot_pilot.py --runs runs/<run> ...            # curves and the lead-time figure
```

Training on the Cerebras Slurm CPU partition (D015 — no CUDA device is reachable):

```bash
cbrun -t rocky -- srun -c 16 -- python -m swe_sr.train \
    --config configs/model/unet_x4.yaml --experiment configs/experiment/full.yaml
```

Measured there: ~0.95 h for EDSR and ~1.65 h for U-Net at 30,000 steps, BF16 on AMD EPYC 9R14.

## Documentation

Read in this order. `CLAUDE.md` lists the same set and states the project's non-negotiable rules.

| File | Contents |
|---|---|
| `docs/PROJECT_SPEC.md` | goal, scope, research questions, definition of done, open questions |
| `docs/DATASET.md` | resolution pairs, IC family, splits, normalization, storage contract |
| `docs/ARCHITECTURE.md` | repository layout, batch contract, both model designs, run directory |
| `docs/EXPERIMENT_PLAN.md` | baselines, loss, optimizer schedule, staged runs, artifacts |
| `docs/VALIDATION.md` | data gates, metric definitions, aggregation protocol, negative tests |
| `docs/DECISIONS.md` | every project decision, D001 to D020, with reasons and consequences |
| `docs/EXPERIMENT_FREEZE.md` | the frozen T-03 record: hashes, commits, seeds, checkpoint digests |
| `docs/RESULTS.md` | generated comparison, lead-time breakdown, fresh workloads, limitations |
| `docs/SIGNOFF.md` | the I-03 audit: what is verified, what is outstanding |
| `TASKS.md` | task status and per-gate evidence |

## Verification posture

Numbers here are recomputed independently before being reported.
`scripts/verify_independent.py` reimplements normalization, destaggering, every field metric, the
mass diagnostic, and the aggregation protocol in plain numpy from the specification, importing
only the HDF5 reader and the model definition. It found a real defect that way — an evaluation
metric whose value depended on batch size — which reusing the package's own metrics could not
have caught.

`docs/SIGNOFF.md` records what is verified and what is not, including known limitations.
