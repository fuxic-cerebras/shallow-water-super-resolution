# Implementation Tasks

Lifecycle: `unclaimed -> in-progress -> ready-for-review -> verified -> complete`.
Only the Integration Lead marks `complete`. Every handoff follows
`docs/AGENT_WORKFLOW.md`.

## Research and contracts

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| R-01 | ready-for-review | Research | - | Audit papers and all four reference repos; pin commits/licenses and build a design-delta matrix | G0 |
| I-01 | ready-for-review | Lead | R-01 | Freeze channel order, generic x4 batch contract, metrics, and pair IDs | G0 |

G0 note: R-01 and I-01 are `ready-for-review`, not `complete`. Pinned commits and
licenses for all four repositories are in `docs/REFERENCES.md`, the design-delta rows
are verified with evidence in `docs/RESEARCH_MATRIX.md`, and contracts are frozen as
D010-D015 in `docs/DECISIONS.md`. Unverified scope, stated rather than waived: the
three source papers were not re-audited against their arXiv originals, and
`climatereconstructionAI` was taken on its documentation instead of a code audit
because no code from it is on the critical path.

## Foundation and solver

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| F-01 | ready-for-review | Lead | I-01 | Add modern project definition, locked environment, lint/type/test config, and CPU CI | G1 |
| P-01 | ready-for-review | PDE/Data | F-01 | Capture deterministic `swe.py` regression fixtures | G1 |
| P-02 | ready-for-review | PDE/Data | P-01 | Extract typed config/state, analytic ICs, and headless `solve()` while preserving demo behavior | G1 |
| P-03 | ready-for-review | PDE/Data | P-02 | Add actual CFL, finite, positive-depth, wall, and mass diagnostics | G1 |

G1 acceptance: deterministic per-field parity within documented tolerance; no plotting
import in generation; all physical diagnostics pass.

G1 evidence: parity is bit-for-bit rather than within a tolerance, since the kernel is a
transcription of the pinned script. 59 tests pass; ruff, ruff format, and strict mypy are
clean. Headlessness is enforced in a subprocess so an already-imported matplotlib cannot
mask a real dependency. Measured: relative mass drift below 1e-13 over a full 3336-step
trajectory, total depth strictly positive, wall-normal velocity exactly zero, and gravity
CFL 0.100 at the fine grid. A divergent configuration is exercised as a negative control
so the diagnostics are not vacuously green.

## Paired data

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| D-01 | ready-for-review | PDE/Data | P-03 | Create immutable 48-IC registry with stable IDs and 32/8/8 split | G2 |
| D-02 | ready-for-review | PDE/Data | D-01 | Implement primary and backup configs plus independent LR/HR paired generation | G2 |
| D-03 | ready-for-review | PDE/Data | D-02 | Add coordinate/time hashes, streaming storage, immutable manifests, checksums | G2 |
| D-04 | unclaimed | PDE/Data | D-03 | Compute separate train-only normalization per pair and full-frame vector augmentation | G2 |
| V-01 | unclaimed | Verifier | D-04 | Independently audit both smoke datasets, including negative leakage/alignment tests | G2 |
| D-05 | unclaimed | PDE/Data | V-01 | Generate and release full primary dataset | G3 |
| D-06 | unclaimed | PDE/Data | D-05 | Benchmark, then stream and release full backup dataset without delaying primary | G3 |
| V-02 | unclaimed | Verifier | D-05,D-06 | Recompute splits, checksums, times, coordinates, normalization, and diagnostics | G3 |

G2 evidence so far: D-01 to D-03 are `ready-for-review`. Both smoke pairs generate with
zero checksum mismatches, disjoint splits, bit-identical within-pair saved times, and mass
drift at roundoff. The new 197-snapshot cadence (D017) was additionally validated at full
scale: all 48 primary trajectories generated in 2m24s, 0 checksum mismatches, worst mass
drift 1.07e-14, minimum depth 98.669 m, 1.41 GiB on disk under gzip. That run is a cadence
validation, NOT the D-05 release: its manifest records a `-dirty` commit, so it is not
reproducible from a commit alone, and D-05 still depends on D-04 normalization and the
V-01 independent audit.

G2 acceptance: both smoke pairs independently integrate matching ICs over matching
domains and exact within-pair saved times. G3 acceptance: immutable manifests pass the
independent audit. Primary and backup have separate G3 releases.

## Models and training

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| M-01 | unclaimed | ML | I-01 | Implement nearest and endpoint-aligned bicubic baselines | G4 |
| M-02 | unclaimed | ML | R-01,I-01 | Implement clean 2D residual U-Net x4 from documented adaptations | G4 |
| M-03 | unclaimed | ML | R-01,I-01 | Adapt EDSR x4 to three normalized physical channels | G4 |
| M-04 | unclaimed | ML | M-02,M-03 | Add generic 32->128 and 64->256 shape/gradient/reload tests and resource metrics | G4 |
| V-03 | unclaimed | Verifier | M-01,M-04 | Independently audit model contracts, baselines, gradients, and checkpoint round-trip | G4 |
| T-01 | unclaimed | ML | V-03,D-05 | Implement normalized MSE, validation, best/last checkpoints, early stopping, provenance | G5 |
| T-02 | unclaimed | ML | T-01 | Run deterministic smoke and primary pilot for both models; project runtime/memory | G5 |
| V-04 | unclaimed | Verifier | T-02 | Audit training determinism, metric path, checkpoint selection, and budget | G5 |
| I-02 | unclaimed | Lead | V-02,V-04 | Freeze primary manifest, commit, configs, seed, metrics, and checkpoint rule | G6 |
| T-03 | unclaimed | ML | I-02 | Run full primary U-Net and EDSR training and loss curves | G6 |

Backup 64->256 training is a later scaling task. It uses separate configs, checkpoints,
normalization, and a newly approved compute budget.

## Evaluation and final review

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| E-01 | unclaimed | ML | T-03 | Evaluate baselines/models on all held-out primary test trajectories | G7 |
| E-02 | unclaimed | PDE/Data | I-02 | Generate frozen `fresh_id` and `ring_ood` evaluation workloads | G7 |
| E-03 | unclaimed | ML | E-01,E-02 | Evaluate fresh workloads without tuning and create tables/plots | G7 |
| V-05 | unclaimed | Verifier | E-03 | Recompute selected metrics and audit leakage, units, aggregation, and claims | G7 |
| I-03 | unclaimed | Lead | V-05 | Clean-install reproduction, documentation audit, integration and science sign-off | G8 |

## Optional follow-ups

- O-01: train both models separately on the backup 64->256 release.
- O-02: training-trajectory-count ablation.
- O-03: single-field versus three-field ablation.
- O-04: direct prediction versus bicubic residual.
- O-05: separately approved physics-loss experiment.
- O-06: continuous space-time or dynamic solver-coupling proposal.
