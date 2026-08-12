# Project Specification

## Goal

Build and compare two neural networks that reconstruct a 128 x 128 shallow-water
state from a 32 x 32 state over the same physical domain. Also produce a separately
versioned 64 x 64 -> 256 x 256 backup dataset for a later scaling study:

$$
F_\theta: \mathbb{R}^{3\times32\times32}
\rightarrow \mathbb{R}^{3\times128\times128}.
$$

The channels are surface elevation and depth-averaged velocity,
`[eta, u, v]`. The project tests whether a neural surrogate can recover spatial
detail lost by a coarse PDE discretization while retaining physically meaningful
large-scale behavior.

## Scope

Version 1 includes:

- refactoring `swe.py` into a headless, importable solver without changing its
  numerical scheme;
- paired coarse/fine data generation from varied analytic initial conditions;
- primary 32 -> 128 and backup 64 -> 256 paired releases with a shared
  initial-condition registry but separate manifests and normalization;
- trajectory-level train, validation, and test splits;
- a residual U-Net and an EDSR baseline, both with x4 spatial output;
- normalized MSE training and validation curves;
- full-frame held-out evaluation against nearest-neighbor and bicubic baselines;
- a fresh post-training solver workload, including an out-of-distribution ring wave;
- reproducible configs, manifests, checkpoints, tables, and plots.

Version 1 does not include:

- temporal forecasting;
- coupling the network into the solver time-stepping loop;
- learned closure or source terms;
- unstructured meshes;
- adversarial or perceptual image losses;
- claims of physical generalization beyond tested initial-condition families.

## Research questions

1. Can either network beat bicubic interpolation on held-out coarse-solver states?
2. Does U-Net's multi-scale encoder/decoder outperform EDSR's residual stack for
   `[eta, u, v]` fields?
3. Are gains consistent across all three channels rather than dominated by `eta`?
4. Does lower pixel MSE also improve mass, energy, and boundary-condition errors?
5. How sharply does performance degrade on a newly generated ring-wave workload?

## Definition of done

The first study is complete when:

- raw and processed dataset manifests reproduce all 48 paired trajectories;
- split auditing proves that no trajectory appears in more than one split;
- U-Net and EDSR each produce a best-validation checkpoint and train/validation
  MSE curve;
- both checkpoints are evaluated on all 1,024 held-out test snapshots;
- both checkpoints are evaluated on a workload generated after training;
- nearest-neighbor and bicubic results use the identical test states and metrics;
- a machine-readable metrics file and a human-readable comparison report exist;
- commands and environment metadata are sufficient to reproduce every result.

Beating a baseline is a research outcome, not a condition for honestly completing
the experiment.

## Constraints

- Full data generation plus both primary training runs should fit within one day on
  a single modern 24 GB or larger CUDA GPU and a multi-core CPU. This is a budget,
  not a hardware-independent runtime promise.
- Raw dataset target: less than 2 GB in float32 before optional compression.
- Default training ceiling: 30,000 optimizer steps per model, with early stopping.
- The one-day and less-than-2-GiB targets apply to the primary study. Full backup
  generation and training receive a separate measured budget.
- All stored raw fields remain float32 in SI units.
- All random sources and split manifests are explicitly seeded.

## Open questions

- Which target training device should define the final runtime budget?
- Should a later study vary physical parameters (`H`, `f_0`, `beta`) as well as
  initial conditions?
- Should version 2 add a physics-weighted auxiliary loss after the MSE-only baseline
  is established?
- Should the solver move from wall boundaries to periodic boundaries for turbulence
  studies, or remain faithful to the current closed-basin workload?
