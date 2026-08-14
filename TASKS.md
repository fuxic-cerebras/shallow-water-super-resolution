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
| D-04 | ready-for-review | PDE/Data | D-03 | Compute separate train-only normalization per pair and full-frame vector augmentation | G2 |
| V-01 | ready-for-review | Verifier | D-04 | Independently audit both smoke datasets, including negative leakage/alignment tests | G2 |
| D-05 | ready-for-review | PDE/Data | V-01 | Generate and release full primary dataset | G3 |
| D-06 | deferred | PDE/Data | D-05 | Benchmark, then stream and release full backup dataset without delaying primary | G3 |
| V-02 | ready-for-review | Verifier | D-05 | Recompute splits, checksums, times, coordinates, normalization, and diagnostics (primary only while D-06 is deferred) | G3 |

G2 evidence so far: D-01 to D-04 are `ready-for-review`. Both smoke pairs generate with
zero checksum mismatches, disjoint splits, bit-identical within-pair saved times, and mass
drift at roundoff. The new 197-snapshot cadence (D017) was additionally validated at full
scale: all 48 primary trajectories generated in 2m24s, 0 checksum mismatches, worst mass
drift 1.07e-14, minimum depth 98.669 m, 1.41 GiB on disk under gzip. That run is a cadence
validation, NOT the D-05 release: its manifest records a `-dirty` commit, so it is not
reproducible from a commit alone, and D-05 still depends on D-04 normalization and the
V-01 independent audit.

G3 evidence (primary only; backup deferred): the full 48-trajectory primary release is
generated with clean provenance at commit `c82ce16` and passes all 12 gates of
`python -m swe_sr.data.validate`, including 384 array checksums recomputed from the arrays
and a train-only normalization recomputed and matched. Worst relative mass drift 1.067e-14,
minimum total depth 98.6689 m, 197 frames over 34.86 h, splits disjoint across 48 unique
IDs.

PROMOTED 2026-08-13: the release now sits at the canonical
`data/{raw,processed}/swe_gaussian_32x128_v1/`. Frozen manifest hash re-verified unchanged at
`af02e44f...`, validation passes 12/12 there, and both frozen runs re-evaluate to identical
numbers. The superseded development dataset was moved aside rather than deleted, and
`data/staging/` holds symlinks so the frozen runs' recorded manifest paths stay resolvable
without editing their artifacts. See the promotion section of `docs/EXPERIMENT_FREEZE.md`.

G2 acceptance: both smoke pairs independently integrate matching ICs over matching
domains and exact within-pair saved times. G3 acceptance: immutable manifests pass the
independent audit. Primary and backup have separate G3 releases.

## Models and training

G4 evidence: M-01 to M-04 and V-03 are `ready-for-review`. 42 model tests and 16 metric
tests pass. Both models take one set of weights to `[B,3,32,32] -> [B,3,128,128]` and
`[B,3,64,64] -> [B,3,256,256]`, every trainable parameter receives a finite gradient at both
resolutions, and checkpoint reload reproduces inference bitwise. U-Net 1,930,208 parameters
(7.72 MB), EDSR 1,517,571 (6.07 MB) -- close enough that the comparison is about architecture
rather than capacity. Zeroing all weights reproduces the bicubic baseline exactly, which is
what makes any reported gain over bicubic attributable to the learned residual.

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| M-01 | ready-for-review | ML | I-01 | Implement nearest and endpoint-aligned bicubic baselines | G4 |
| M-02 | ready-for-review | ML | R-01,I-01 | Implement clean 2D residual U-Net x4 from documented adaptations | G4 |
| M-03 | ready-for-review | ML | R-01,I-01 | Adapt EDSR x4 to three normalized physical channels | G4 |
| M-04 | ready-for-review | ML | M-02,M-03 | Add generic 32->128 and 64->256 shape/gradient/reload tests and resource metrics | G4 |
| V-03 | ready-for-review | Verifier | M-01,M-04 | Independently audit model contracts, baselines, gradients, and checkpoint round-trip | G4 |
| T-01 | ready-for-review | ML | V-03,D-05 | Implement normalized MSE, validation, best/last checkpoints, early stopping, provenance | G5 |
| T-02 | ready-for-review | ML | T-01 | Run deterministic smoke and primary pilot for both models; project runtime/memory | G5 |
| V-04 | unclaimed | Verifier | T-02 | Audit training determinism, metric path, checkpoint selection, and budget | G5 |
| I-02 | complete | Lead | V-02,V-04 | Freeze primary manifest, commit, configs, seed, metrics, and checkpoint rule | G6 |
| T-03 | ready-for-review | ML | I-02 | Run full primary U-Net and EDSR training and loss curves | G6 |
| M-05 | in-progress | ML | I-02 | Add ConvMixer x4 as a third architecture and run it on the frozen schedule (D023) | G6 |

G5 evidence (measured on Slurm jobs 295533 and 295561, node cpu-dy-x48-m7a-3, 16 threads,
BF16 on AMD EPYC 9R14): both pilots ran 1,970 steps over 10 epochs with validation loss
falling monotonically. EDSR 0.1134 s/step, 515 MB peak RSS, best validation 0.306881. U-Net
0.1979 s/step, 813 MB, best 0.302805. Measured 4.6x speedup over 2 local threads. Projected
full 30,000-step runs: EDSR 0.95 h, U-Net 1.65 h, 2.59 h combined, well inside the 8 h per
model envelope in `docs/EXPERIMENT_PLAN.md`.

LEAD-TIME FINDING (revised after the full diagnostic run; two earlier readings retracted).

The durable fact: the independent coarse and fine integrations D002 requires progressively
decorrelate, so baseline error grows monotonically with lead time. Measured on the full
eight-trajectory validation split, bicubic goes 0.008 at 2.0 h, 0.105 at 8.0 h, 0.442 at
18.1 h, 1.117 at 34.2 h. Beyond roughly 30 h the coarse state carries little information
about the specific fine realization, so error there is not recoverable by any method. Any
single aggregate therefore summarizes a mixture of a well-posed and a partly unpredictable
regime, which is why every report stratifies by lead time.

RETRACTED #1: an earlier note here claimed bicubic scores 1.019, equal to predicting the
channel mean. That came from two of eight validation trajectories. The full-split value is
0.468, median 0.341, trajectory-level 95% CI [0.220, 0.736]. Bicubic reaches mean-predictor
level only beyond about 30 h, not overall.

RETRACTED #2: an earlier note claimed full training does not improve short-lead-time error,
read off a log-scale plot at epoch 11 of 38. The completed 30,000-step EDSR run shows the
opposite. Improvement from epoch 1 to final was largest at the shortest lead time: 0.2502 to
0.0535 at 2.0 h, a 78.6 percent reduction, against a mean of 42.9 percent across all lead
times. The crossover where the model overtakes bicubic moved from about 9 h to about 6 h. A
residual deficit below 6 h remains real -- 0.0535 against bicubic's 0.0080 at 2.0 h -- but it
narrowed from 31x to 6.7x and was still falling when the step cap hit.

Consequence for the pending decision: conditioning the model on lead time is now an
optimization rather than a fix for a broken objective. Stratified reporting appears
sufficient. Still awaiting an owner decision before I-02 freezes anything.

Backup 64->256 training is a later scaling task. It uses separate configs, checkpoints,
normalization, and a newly approved compute budget.

Backup work is DEFERRED as of 2026-08-12, by decision of the project owner, to keep the
primary 32->128 experiment moving. This is the ordering `docs/AGENT_WORKFLOW.md` already
prescribes: backup data is a separate G3 release that must not delay the primary. Deferred
items: D-06 (full backup release), the backup half of V-02, and O-01 (backup training).
Their tests are marked `backup` rather than deleted, so coverage returns with
`pytest -m backup`; `scripts/check.sh` deselects that marker and echoes the exclusion.
Backup *config* assertions stay in the default suite, since they cost nothing and pin the
D017 stride and duration relationship. Nothing about the backup design is withdrawn: its
config, smoke config, and the D008/D017 decisions all stand.

G6 FROZEN 2026-08-13 by owner authorization. I-02 is complete: the experiment is frozen in
`docs/EXPERIMENT_FREEZE.md` and recorded as D020, pinning the dataset and manifest hashes, IC
registry hash, launch commits, config hashes, seed, normalization statistics, metric
definitions, checkpoint rule, checkpoint SHA-256 digests, and the held-out test results.

T-03 results, held-out test split, 1,576 snapshots over 8 trajectories, both independently
recomputed from arrays with 20 checks each passing:

| method | params | normMSE | 95% CI |
|---|---:|---:|---|
| nearest | 0 | 0.4301 | [0.3076, 0.5435] |
| bicubic | 0 | 0.4295 | [0.3069, 0.5431] |
| EDSR | 1,517,571 | 0.0830 | [0.0543, 0.1129] |
| U-Net | 1,930,208 | 0.0400 | [0.0261, 0.0544] |

U-Net wins by 2.1x. Two findings worth carrying into any write-up: lower pixel MSE does not
imply better physics, since EDSR worsens relative mass error to 0.0540 against bicubic's 0.0392
while beating it fivefold on MSE; and the early-lead-time deficit that once looked like a
broken objective is absent for U-Net, which matches bicubic at 2 h.

Note the two runs keep `stage: diagnostic` in their own artifacts, deliberately. They were
launched as diagnostics and a completed run's provenance is not rewritten to match a later
decision. `swe_sr.report` reads the freeze record, so it identifies them as T-03 correctly
without altering them.

## Ablations

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| A-01 | ready-for-review | ML | I-02 | Ablation 3: outer bicubic residual against direct prediction, both architectures (D022) | - |
| A-02 | unclaimed | ML | M-05 | Follow-up: run the D022 direct arm for ConvMixer (`outer_baseline: none`); the flag works but no config or run ships | - |
| A-03 | ready-for-review | ML | M-05 | ConvMixer normalization: BatchNorm as published against EDSR's unnormalized recipe, the one place D023 knowingly contradicts EDSR | - |
| A-04 | unclaimed | ML | M-05 | Follow-up: ConvMixer `patch_size` > 1, a one-line config change the paper predicts should hurt at 32x32 | - |
| A-05 | superseded | ML | M-05 | Regularization for ConvMixer. Withdrawn: the pilot's 8.1x train/val gap was its 8-trajectory subset, and the full run's gap is 1.79x, matching both other models. The A-03 arm underfits rather than overfits, so regularization is the wrong lever | - |
| A-06 | unclaimed | ML | A-03 | Follow-up: sweep `res_scale` for the unnormalized arm. It was set to EDSR's published 0.1 untuned, and since it both enables training and damps every block it may account for part of A-03's 1.72x gap | - |
| A-07 | unclaimed | ML | A-03 | Follow-up: pre-activation unnormalized ConvMixer, i.e. `x + s*DW(GELU(x))` so the residual branch ends linear as EDSR's does. Measured at init this removes the DC drift entirely (+0.002 against +0.365 after 16 blocks), so it separates "ConvMixer needs normalization" from "ConvMixer needs mean-centering" -- the more interesting question than A-06 | - |

A-03 evidence (2026-08-13, owner-requested; write-up in `docs/ABLATION_NORMALIZATION.md`). Tests
whether EDSR's no-normalization finding transfers to ConvMixer. It does **not**. Held-out test
split, paired on trajectory, positive favours BatchNorm:

| arm | test MSE | 95% CI | paired diff | 95% CI | BatchNorm wins |
|---|---:|---|---:|---|---|
| BatchNorm as published | 0.0651 | [0.0366, 0.0963] | +0.04692 | [+0.01308, +0.08075] | 7 of 8 |
| EDSR-style unnormalized | 0.1120 | [0.0772, 0.1469] | | | |

Not a single-factor ablation, and it cannot be: removing BatchNorm alone is untrainable, because
ConvMixer's pointwise stage is non-residual and the first block's gradient norm falls to 1.1e-11.
The arm therefore adopts EDSR's whole recipe (residual everywhere, `res_scale=0.1`,
unnormalized). BatchNorm winning is the unambiguous direction: it beat a design built to work
without it. The unnormalized arm **underfits** — final train loss 0.1082 against 0.0103 — so
BatchNorm's benefit here is optimization, not regularization. Both runs pass
`scripts/verify_independent.py`.

A-01 evidence (2026-08-13, owner-requested). Both architectures retrained from scratch with the
additive bicubic path removed, on the frozen manifest, frozen seed 20260812, and a schedule
identical in every value to the frozen runs'. Parameter counts identical between arms, so exactly
one factor varies. Both new runs pass `scripts/verify_independent.py`. Frozen T-03 artifacts and
`docs/RESULTS.md` untouched; results in `docs/ABLATION_RESIDUAL.md`.

Paired on trajectory, held-out test split, normalized macro MSE (negative favours direct):

| arch | residual | direct | paired diff | 95% CI | excludes 0 | direct wins |
|---|---:|---:|---:|---|---|---|
| EDSR | 0.0830 | 0.0813 | -0.00166 | [-0.00315, -0.00033] | yes | 6 of 8 |
| U-Net | 0.0400 | 0.0421 | +0.00212 | [-0.00023, +0.00504] | no | 3 of 8 |

Direct prediction is better for EDSR by a paired 2.0% relative, and indistinguishable for the
U-Net. Two secondary findings matter more than the headline: the gain is NOT at short lead time
where the hypothesis predicted it -- both direct arms are slightly worse at 2 h, and EDSR's gain
appears from 4.7 h onward -- and all four runs still lose to bicubic at 2 h, so the outer form is
not the cause of that deficit and the objective remains the suspect. The crossover differs sharply
by architecture: EDSR is behind out to between 4.7 and 7.4 h, the U-Net only to about 2.4 h.
One seed per arm; treat as evidence on this split rather than an established property.

## Evaluation and final review

| ID | Status | Owner | Depends | Task | Gate |
|---|---|---|---|---|---|
| E-01 | ready-for-review | ML | T-03 | Evaluate baselines/models on all held-out primary test trajectories | G7 |
| E-02 | ready-for-review | PDE/Data | I-02 | Generate frozen `fresh_id` and `ring_ood` evaluation workloads | G7 |
| E-03 | ready-for-review | ML | E-01,E-02 | Evaluate fresh workloads without tuning and create tables/plots | G7 |
| V-05 | ready-for-review | Verifier | E-03 | Recompute selected metrics and audit leakage, units, aggregation, and claims | G7 |
| I-03 | signed-off-with-exceptions | Lead | V-05 | Clean-install reproduction, documentation audit, integration and science sign-off | G8 |

G8 evidence, clean-install reproduction (verified from a fresh clone of the repository with
`--recurse-submodules`, a fresh venv, and `pip install -e ".[dev]"`):

- all six entry points documented in `CLAUDE.md` exist and respond;
- `scripts/check.sh` exits 0 from the clean install: 309 tests, ruff, ruff format, strict mypy;
- `python -m swe_sr.data.generate --config configs/data/primary_32x128_smoke.yaml` runs
  end-to-end and reproduces the **identical** IC registry hash
  `976e3a577a25a633c6a2625263f23e60482768965029805a5efd16be97ab7c8c`, so the dataset is
  reproducible from source rather than merely regenerable;
- `python -m swe_sr.data.validate` passes 12/12 gates on what that produced.

Limitation stated rather than glossed: the venv used `--system-site-packages` so torch was
reused rather than downloaded, since a full fresh torch install is a 200 MB download that CI
already covers. Packaging, imports, entry points, and the documented commands were exercised
fresh; only the torch wheel resolution was not.

I-03 remains `in-progress`, not complete. Sign-off requires the results it would sign off on,
and those depend on the outstanding lead-time decision recorded under G5.

G8 SIGN-OFF 2026-08-13. `docs/SIGNOFF.md` records the I-03 audit: all eight definition-of-done
items pass, all five research questions are answered from measured results, both frozen
checkpoints are independently recomputed with 20 checks each passing, and six findings that
revise the original specification are recorded as decisions.

Three exceptions are stated rather than waived: the frozen dataset sits at `data/staging/` rather
than the canonical path pending owner approval to remove a stale directory; the backup 64->256
pair is deferred; and clean-install reproduction did not re-resolve the torch wheel locally,
though CI covers it.

Sign-off scope limit, stated plainly in `docs/SIGNOFF.md`: a single agent performed every role, so
the independence the G7 and G8 gates assume was not available. Tasks therefore remain
`ready-for-review` rather than `complete`, apart from owner-authorized I-02. The document names
the three areas most deserving external review: D011 destaggering and its effect on reported
velocity errors, the aggregation protocol implementation given a batch-dependence defect was
already found there, and the D018 claim that the specified augmentation is invalid.

## Optional follow-ups

- O-07: **ready-for-review** — cross-resolution transfer, run 2026-08-13 on owner request. Generate
  the 64->256 pair and evaluate the frozen 32->128 checkpoints on it with no retraining (D021,
  `docs/TRANSFER.md`). Evidence: 12/12 data gates at the new pair; both models still beat bicubic
  but skill falls from 5.2x to 2.4x (EDSR) and 10.7x to 2.2x (U-Net); the ranking inverts, EDSR
  0.0444 against U-Net 0.0497, paired `+0.00533 [+0.00268, +0.00826]` excluding zero with U-Net
  worse on 8 of 8 trajectories; both models lose to bicubic below about 12.7 h, and the exact
  `r^2 - 2rc + 1` decomposition attributes that to correction magnitude (r up to 9.5) rather than
  direction.
- O-01: train both models separately on the backup 64->256 release. **Still deferred** — the data
  now exists as a by-product of O-07, but no training run has used it, which is the whole content
  of O-01.
- O-02: training-trajectory-count ablation.
- O-03: single-field versus three-field ablation.
- O-04: direct prediction versus bicubic residual.
- O-05: separately approved physics-loss experiment.
- O-06: continuous space-time or dynamic solver-coupling proposal.
