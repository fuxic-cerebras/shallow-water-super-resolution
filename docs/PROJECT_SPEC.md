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
- both checkpoints are evaluated on all 1,576 held-out test snapshots;
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

- ~~Which target training device should define the final runtime budget?~~ Answered by
  D015: the Slurm `cpu` partition at `-c 16` in BF16. No CUDA device is reachable.
- Should a later study vary physical parameters (`H`, `f_0`, `beta`) as well as
  initial conditions?
- Should version 2 add a physics-weighted auxiliary loss after the MSE-only baseline
  is established?
- Should the solver move from wall boundaries to periodic boundaries for turbulence
  studies, or remain faithful to the current closed-basin workload?

### Discretization ambiguities found during G0

These are recorded rather than resolved. None blocks version 1, but each bounds what
the results can claim.

- **Wall position is ambiguous in the reference solver.** Under a strict Arakawa
  C-grid reading of `MODEL_NOTES.md`, the no-flow walls sit at `+/-(L/2 + dx/2)`,
  which makes the 32-point basin 2.4 percent wider than the 128-point basin
  (1.0323e6 m against 1.0079e6 m). `MODEL_NOTES.md` instead asserts walls at
  `+/-L/2`. D011 sidesteps the consequence by placing every channel on the `eta`
  grid, whose extent is resolution independent, but the underlying discretization is
  genuinely ambiguous and the LR and HR runs are therefore not solving on exactly the
  same basin.
- **Saved times differ across resolution pairs by 0.392 percent.** The saved interval
  is 603.355 s for the primary pair and 600.989 s for the backup pair, because the
  spacing ratio is `127/255` rather than exactly `1/2`. Within a pair, LR and HR saved
  times are bit-identical as `docs/DATASET.md` requires. The paired cross-resolution
  bootstrapping described in `docs/DATASET.md` therefore compares states at slightly
  different physical times, which must be stated wherever such a comparison is
  reported.
- **Wide initial conditions are center-biased.** The `2 sigma` wall margin with
  `sigma` up to 120 km confines bump centers to `+/-260 km` of a `+/-500 km` domain,
  so wide bumps cannot be placed off-center and the realized center distribution is
  not uniform over the domain. Record the realized `sigma/dx` and center
  distributions in the IC registry so the coverage is auditable.
