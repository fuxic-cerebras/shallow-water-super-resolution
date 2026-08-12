# Experiment Plan

## Baselines

Evaluate the following on identical full-frame states:

1. Nearest-neighbor x4 interpolation.
2. Bicubic x4 interpolation.
3. Residual U-Net x4.
4. EDSR x4.

Run the first complete experiment on `swe_gaussian_32x128_v1`. Generate and validate
`swe_gaussian_64x256_v1` now, but train it only as a later scaling experiment. The
same resolution-generic model code uses separate configs, normalization, checkpoints,
and batch sizes for the two pair IDs.

Interpolation operates channel by channel in normalized coordinates, followed by
de-normalization for physical metrics. Record interpolation settings such as
`align_corners` so they are identical in training residuals and evaluation.

## Loss

Use equal-weight normalized channel MSE as the version 1 optimization objective:

$$
\mathcal{L}_{MSE} = \frac{1}{3}\sum_{c\in\{\eta,u,v\}}
\frac{1}{N_c}\sum_i(\hat{y}'_{c,i}-y'_{c,i})^2.
$$

This follows the squared L2 reconstruction objective used by Fukami et al. while
channel normalization prevents the field with the largest numerical scale from
dominating. Do not add physics, gradient, spectral, perceptual, or adversarial terms
until the MSE-only comparison is complete.

## Default optimizer schedule

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| Weight decay | 1e-6 |
| Scheduler | cosine decay with 500-step warm-up |
| Precision | BF16 if supported, otherwise FP16/FP32 |
| Training sample | Full 32 x 32 LR -> 128 x 128 HR frame (primary) |
| Batch size | 8, increased only after the pilot |
| Gradient clipping | global norm 1.0 |
| Maximum epochs | 100 |
| Maximum optimizer steps | 30,000 |
| Early stopping | 15 validations without improvement |
| Checkpoint criterion | lowest full validation MSE |
| Primary seed | 20260812 |

Validate on the complete 1,024-frame validation split after every epoch. Save train
MSE, validation MSE, per-channel validation MSE, learning rate, throughput, elapsed
time, and peak accelerator memory.

## Staged runs

### Smoke

- Two trajectories and eight frames.
- Two training steps per model.
- Goal: shapes, gradients, checkpoints, evaluation, and plots all work.

### Pilot

- Eight training trajectories, two validation trajectories.
- Ten epochs or 2,000 steps.
- Goal: estimate memory and wall time, verify falling validation loss, and choose a
  safe batch size without changing the architecture.

### Full

- Fixed 32/8/8 trajectory split.
- Defaults above, with early stopping.
- Run U-Net and EDSR with the same data manifest and primary seed.
- If either run is projected to exceed 10 hours, cap both models at the same number
  of optimizer steps and record the budget adjustment before running.

The intended one-day envelope applies to the primary study: up to 4 hours for data generation and validation,
up to 8 hours per model, and up to 4 hours for final evaluation/reporting. Actual
timings must replace these planning numbers in the report.

## Curves and artifacts

Produce:

- train and validation MSE versus optimizer step;
- train and validation MSE versus wall time;
- per-channel validation MSE versus epoch;
- a final table for all baselines and models;
- at least four fixed qualitative test examples showing LR bicubic, prediction, HR,
  and signed error for each field with shared color limits;
- the same artifacts for `fresh_id` and `ring_ood`, clearly labeled as fresh tests.

## Ablations after the primary comparison

Run only after the two primary models are complete:

1. Training-set size: 8, 16, and 32 trajectories.
2. `eta` only versus all three coupled fields.
3. Outer bicubic residual versus direct prediction.
4. Optional physics loss for mass or shallow-water residuals.

Each ablation changes one factor and reuses the fixed validation/test manifests.
