# Validation and Evaluation

## Data validation gates

Before training, fail if any of these checks fails:

- manifest schema, config hash, and array checksums are valid;
- LR and HR shapes match the node counts and x4 shape contract declared by the
  manifest for every trajectory;
- LR and HR coordinates cover the same physical domain;
- LR and HR saved-time arrays are exactly equal;
- every value is finite and every `H + eta` value is positive;
- trajectory IDs and initial-condition seeds are disjoint across splits;
- recomputed training normalization matches the manifest;
- solver mass drift is reported and remains within the tolerance established by the
  solver regression tests.

## Unit and integration tests

Minimum fast test suite:

- current `swe.py` default run agrees with the refactored numerical kernel for a
  short deterministic case;
- identical configs and seeds reproduce identical arrays and manifests;
- coarse and fine initial states are analytic evaluations of the same specification;
- paired samples have equal physical time and domain metadata;
- normalization round-trips each channel;
- vector-aware flips change the correct velocity sign;
- U-Net and EDSR return `[B, 3, 128, 128]` from `[B, 3, 32, 32]`;
- U-Net and EDSR also return `[B, 3, 256, 256]` from `[B, 3, 64, 64]`;
- both models produce finite gradients for all trainable parameters;
- metric implementations match small hand-computed examples;
- checkpoint reload reproduces inference output.

## Primary metrics

Training and model selection use normalized MSE. Final reports also include physical
metrics after de-normalization.

For each field `c`:

$$
\operatorname{RMSE}_c = \sqrt{\frac{1}{N_c}\sum_i
(\hat{y}_{c,i}-y_{c,i})^2},
$$

$$
\operatorname{relL2}_c =
\frac{\|\hat{y}_c-y_c\|_2}{\|y_c\|_2+10^{-12}}.
$$

Report:

- normalized MSE, macro-averaged across the three channels;
- per-channel normalized MSE;
- per-channel physical RMSE with SI units;
- per-channel relative L2;
- per-channel maximum absolute error;
- sample median and 95th-percentile relative L2.

Macro-average normalized per-channel errors. Never flatten raw physical channels into
one scalar because their units and magnitudes differ.

## Physical diagnostics

Compute these as diagnostics, not optimization claims:

- relative total mass error from `h = H + eta`;
- relative domain-integrated energy error using the project's documented discrete
  convention;
- wall-normal velocity error at all boundaries;
- negative-depth count and minimum predicted depth;
- optional gradient RMSE for each field;
- radially averaged spectra only if the boundary treatment and windowing are stated.

Compare every diagnostic with nearest-neighbor and bicubic interpolation.

## Aggregation protocol

1. Compute metrics per snapshot and per channel.
2. Aggregate over time within each trajectory.
3. Aggregate trajectories with equal weight.
4. Report mean, median, standard deviation, and 95% bootstrap confidence interval
   over trajectories where meaningful.

This prevents a trajectory with more usable frames from dominating the result.

## Fresh-workload protocol

- Freeze normalization, architecture, weights, and model-selection decisions.
- Generate `fresh_id` and `ring_ood` using new manifest IDs.
- Run both models and interpolation baselines without fine-tuning.
- Report fresh results separately from the original test split.
- If out-of-distribution results fail, preserve and report the failure rather than
  changing the workload after inspection.

## Independent verification gates

The Scientific Verifier recomputes checksums, normalization statistics, split
disjointness, and selected metrics from arrays; author logs are not proof. Gates G0-G8
and ownership rules are in `docs/AGENT_WORKFLOW.md`. Negative tests include split
overlap, mismatched saved times, coordinate-hash mismatch, wrong vector reflection,
wrong normalization pair ID, and invalid non-x4 output.

## Comparison table

At minimum, the final report contains:

| Method | Params | Test MSE | eta relL2 | u relL2 | v relL2 | Mass err | Time/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nearest | 0 | | | | | | |
| Bicubic | 0 | | | | | | |
| U-Net x4 | | | | | | | |
| EDSR x4 | | | | | | | |

Empty cells are populated only by generated results. Units and hardware appear in the
caption or adjacent run metadata.
