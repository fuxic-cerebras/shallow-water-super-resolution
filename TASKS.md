# Implementation Tasks

Lifecycle: `unclaimed -> in-progress -> ready-for-review -> verified -> complete`.
Only the Integration Lead marks `complete`. Every handoff follows
`docs/AGENT_WORKFLOW.md`.

## Research and contracts

| ID | Owner | Depends | Task | Gate |
|---|---|---|---|---|
| R-01 | Research | - | Audit papers and all four reference repos; pin commits/licenses and build a design-delta matrix | G0 |
| I-01 | Lead | R-01 | Freeze channel order, generic x4 batch contract, metrics, and pair IDs | G0 |

## Foundation and solver

| ID | Owner | Depends | Task | Gate |
|---|---|---|---|---|
| F-01 | Lead | I-01 | Add modern project definition, locked environment, lint/type/test config, and CPU CI | G1 |
| P-01 | PDE/Data | F-01 | Capture deterministic `swe.py` regression fixtures | G1 |
| P-02 | PDE/Data | P-01 | Extract typed config/state, analytic ICs, and headless `solve()` while preserving demo behavior | G1 |
| P-03 | PDE/Data | P-02 | Add actual CFL, finite, positive-depth, wall, and mass diagnostics | G1 |

G1 acceptance: deterministic per-field parity within documented tolerance; no plotting
import in generation; all physical diagnostics pass.

## Paired data

| ID | Owner | Depends | Task | Gate |
|---|---|---|---|---|
| D-01 | PDE/Data | P-03 | Create immutable 48-IC registry with stable IDs and 32/8/8 split | G2 |
| D-02 | PDE/Data | D-01 | Implement primary and backup configs plus independent LR/HR paired generation | G2 |
| D-03 | PDE/Data | D-02 | Add coordinate/time hashes, streaming storage, immutable manifests, checksums | G2 |
| D-04 | PDE/Data | D-03 | Compute separate train-only normalization per pair and full-frame vector augmentation | G2 |
| V-01 | Verifier | D-04 | Independently audit both smoke datasets, including negative leakage/alignment tests | G2 |
| D-05 | PDE/Data | V-01 | Generate and release full primary dataset | G3 |
| D-06 | PDE/Data | D-05 | Benchmark, then stream and release full backup dataset without delaying primary | G3 |
| V-02 | Verifier | D-05,D-06 | Recompute splits, checksums, times, coordinates, normalization, and diagnostics | G3 |

G2 acceptance: both smoke pairs independently integrate matching ICs over matching
domains and exact within-pair saved times. G3 acceptance: immutable manifests pass the
independent audit. Primary and backup have separate G3 releases.

## Models and training

| ID | Owner | Depends | Task | Gate |
|---|---|---|---|---|
| M-01 | ML | I-01 | Implement nearest and endpoint-aligned bicubic baselines | G4 |
| M-02 | ML | R-01,I-01 | Implement clean 2D residual U-Net x4 from documented adaptations | G4 |
| M-03 | ML | R-01,I-01 | Adapt EDSR x4 to three normalized physical channels | G4 |
| M-04 | ML | M-02,M-03 | Add generic 32->128 and 64->256 shape/gradient/reload tests and resource metrics | G4 |
| V-03 | Verifier | M-01,M-04 | Independently audit model contracts, baselines, gradients, and checkpoint round-trip | G4 |
| T-01 | ML | V-03,D-05 | Implement normalized MSE, validation, best/last checkpoints, early stopping, provenance | G5 |
| T-02 | ML | T-01 | Run deterministic smoke and primary pilot for both models; project runtime/memory | G5 |
| V-04 | Verifier | T-02 | Audit training determinism, metric path, checkpoint selection, and budget | G5 |
| I-02 | Lead | V-02,V-04 | Freeze primary manifest, commit, configs, seed, metrics, and checkpoint rule | G6 |
| T-03 | ML | I-02 | Run full primary U-Net and EDSR training and loss curves | G6 |

Backup 64->256 training is a later scaling task. It uses separate configs, checkpoints,
normalization, and a newly approved compute budget.

## Evaluation and final review

| ID | Owner | Depends | Task | Gate |
|---|---|---|---|---|
| E-01 | ML | T-03 | Evaluate baselines/models on all held-out primary test trajectories | G7 |
| E-02 | PDE/Data | I-02 | Generate frozen `fresh_id` and `ring_ood` evaluation workloads | G7 |
| E-03 | ML | E-01,E-02 | Evaluate fresh workloads without tuning and create tables/plots | G7 |
| V-05 | Verifier | E-03 | Recompute selected metrics and audit leakage, units, aggregation, and claims | G7 |
| I-03 | Lead | V-05 | Clean-install reproduction, documentation audit, integration and science sign-off | G8 |

## Optional follow-ups

- O-01: train both models separately on the backup 64->256 release.
- O-02: training-trajectory-count ablation.
- O-03: single-field versus three-field ablation.
- O-04: direct prediction versus bicubic residual.
- O-05: separately approved physics-loss experiment.
- O-06: continuous space-time or dynamic solver-coupling proposal.
