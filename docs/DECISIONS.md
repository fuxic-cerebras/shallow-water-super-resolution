# Decision Log

## D001 - Three physical channels

- Status: accepted
- Decision: model `[eta, u, v]`; reconstruct total height as `h = H + eta`.
- Reason: `swe.py` evolves `eta`, and removing the constant depth offset improves
  numerical conditioning without discarding information.

## D002 - Independent coarse and fine integrations

- Status: accepted
- Decision: use independent 32/128 primary and 64/256 backup solver pairs over the
  same physical domain.
- Reason: the research question concerns error from a genuinely coarse PDE solve,
  not recovery from an artificially downsampled fine image.

## D003 - Shared fine-grid time step within each pair

- Status: accepted for version 1
- Decision: integrate LR and HR within each resolution pair with that pair's stable
  HR-grid time step. Different pair IDs may use different time steps.
- Reason: exact time alignment avoids interpolation and time-discretization mismatch.
  A later experiment may let each solver use its own stable time step.

## D004 - Trajectory-level split

- Status: accepted
- Decision: split 48 initial-condition trajectories into 32 train, 8 validation, and
  8 test trajectories before simulation.
- Reason: random frame splitting would leak nearly identical adjacent states.

## D005 - MSE-only primary study

- Status: accepted
- Decision: train on equal-weight normalized channel MSE.
- Reason: it directly tests the requested squared L2 objective and makes the U-Net
  versus EDSR comparison interpretable before adding physics losses.

## D006 - Common bicubic residual

- Status: accepted
- Decision: both networks learn a residual over bicubic x4 interpolation.
- Reason: both models start from the same low-frequency baseline, which improves
  optimization and keeps the comparison focused on learned detail.

## D007 - One-day data budget

- Status: accepted as a planning default
- Decision: the primary has 9,456 paired snapshots, approximately 1.84 GiB raw, and
  no more than 30,000 optimizer steps per model. Backup has a separate budget.
- Amended by D017, which raised the snapshot count from 6,144; the storage target and
  the optimizer-step ceiling are unchanged.
- Reason: this is a practical baseline that can be expanded after measured scaling.

## D008 - Dual resolution-pair releases

- Status: accepted
- Decision: release 32 -> 128 as primary and 64 -> 256 as backup, using one immutable
  IC/split registry but separate arrays, manifests, timesteps, normalization, and runs.
- Reason: preserve a fast first experiment while retaining higher-resolution data.

## D009 - Multi-agent ownership and independent verification

- Status: accepted
- Decision: use four specialist subagents plus an integration lead, path ownership,
  explicit handoffs, and independent verification gates.
- Reason: separate PDE/data from ML and prevent authors from self-certifying results.

## D010 - Solver reference pinned as a submodule

- Status: accepted
- Decision: track `fuxic-cerebras/shallow-water` as a Git submodule at
  `references/shallow-water`, pinned at commit
  `a8457df886cec74e2a02652280d2f00de0804dfc`. Do not vendor the files.
- Reason: `CLAUDE.md` names `MODEL_NOTES.md` the source of truth and requires
  preserving `swe.py` behavior, but neither file was in this repository, so P-01 had
  nothing to capture fixtures from. A pinned submodule keeps the reference
  byte-identical, makes provenance exact, and avoids redistributing code whose
  license is unstated (see `docs/REFERENCES.md`).
- Consequences: `swe.py` is *not* importable from the repository root, so the P-01
  parity harness loads it from the submodule path and CI must run
  `git submodule update --init`. Paths in `CLAUDE.md`, `docs/ARCHITECTURE.md`, and
  `.claude/agents/pde-data.md` are updated accordingly.

## D011 - Destagger velocities in the processed layer

- Status: accepted
- Decision: `raw/` stores the solver's faithful staggered output. `processed/`
  reconstructs cell-centered velocities on the `eta` grid,

  ```text
  u_c[0, :] = 0.5 * u_raw[0, :]                    # west wall u = 0, never stored
  u_c[i, :] = 0.5 * (u_raw[i-1, :] + u_raw[i, :])  # i = 1 ... N-1
  ```

  and symmetrically for `v` along y. Models and metrics consume the processed,
  cell-centered fields.
- Reason: `MODEL_NOTES.md` pins an Arakawa C-grid with `u_n[i,j]` at
  `u_{i+1/2,j}` and `v_n[i,j]` at `v_{i,j+1/2}`, while `docs/DATASET.md` stores three
  colocated channels and `docs/ARCHITECTURE.md` mandates endpoint-aligned bicubic.
  That is exact for `eta` and off by half a cell for `u` and `v`. The offset is
  16.1 km on the 32-point grid and 3.9 km on the 128-point grid, so it does not
  cancel between input and target and would be absorbed into the learned residual as
  a systematic spatial shift.
- Consequences: after destaggering all three channels share the `eta` coordinate grid
  `linspace(-L/2, L/2, N)`, whose physical extent is identical at every resolution, so
  endpoint-aligned x4 bicubic with `align_corners=True` is exact for every channel.
  Destaggering is a documented, tested, non-invertible preprocessing step; the raw
  staggered arrays remain the archival record.

## D012 - Wall diagnostics are split across the raw and processed layers

- Status: accepted
- Decision: the exact `u[-1, :] == 0` and `v[:, -1] == 0` checks run against **raw**
  staggered arrays during data validation. The model-facing wall diagnostic compares
  predicted against true cell-centered wall-adjacent velocities.
- Reason: D011 makes `u_c[0, :] = 0.5 * u_raw[0, :]`, which is the correct cell
  average but is not identically zero, so a hard zero assertion is only meaningful on
  the raw arrays.
- Consequences: `docs/VALIDATION.md` says "all boundaries", but only the eastern and
  northern walls carry an explicit stored constraint. The western and southern walls
  are enforced implicitly by the one-sided flux at `swe.py:220` and `swe.py:223`. The
  diagnostic reports east/north as exact-zero checks and west/south as flux checks.

## D013 - HDF5 storage, one file per trajectory

- Status: accepted
- Decision: use chunked HDF5 via `h5py`, one file per trajectory, chunked along the
  time axis.
- Reason: `docs/DATASET.md` permits Zarr or HDF5. `h5py` is already available, gives
  a single stable byte stream per array for checksums, and one file per trajectory
  matches the trajectory-level split. A directory-of-chunks layout would also be slow
  on the NFS workspace.

## D014 - Discrete mass and energy convention

- Status: accepted
- Decision: on the destaggered grid,

  ```text
  M = sum_ij eta * dx * dy
  E = sum_ij [ 0.5 * (H + eta) * (u_c^2 + v_c^2) + 0.5 * g * eta^2 ] * dx * dy
  ```

- Reason: `docs/VALIDATION.md` defers to "the project's documented discrete
  convention", but `MODEL_NOTES.md` never defines one. `M` is the varying part of
  `sum(H + eta)` and matches the `np.sum(eta_n)` diagnostic at `swe.py:255`. `E` is
  the standard shallow-water kinetic-plus-available-potential energy, evaluated at
  cell centers so both velocity components are colocated with `eta`.
- Consequences: energy is a diagnostic only. The scheme is not energy-conserving by
  construction, so a nonzero drift is expected and reported, unlike mass.

## D015 - Training device is the Slurm CPU partition

- Status: accepted
- Decision: run training as `cbrun -t rocky -- srun -c 16 -- <command>` on the Slurm
  `cpu` partition, in BF16. Retain the documented architectures and the 30,000-step
  ceiling unchanged.
- Reason: answers the first open question in `docs/PROJECT_SPEC.md`. No CUDA device is
  available on any reachable partition (`GRES` is null on both `cpu` and `cpu-spot`),
  so the "24 GB or larger CUDA GPU" planning assumption is superseded. The nodes are
  AMD EPYC 9R14 with `avx512_bf16` and `avx512_vnni`, which makes BF16 CPU training
  viable within the documented per-model envelope.
- Consequences: no training run is launched without notifying the project owner and
  receiving explicit authorization. Escalation if the measured pilot overruns the
  envelope, in order: request `-c 48` or `-c 96`, then constrain placement to Intel
  `m7i` nodes for AMX-BF16. Any resulting budget change is recorded here before
  running.

## D016 - I-01 frozen contract

- Status: accepted
- Decision: the following are frozen at G0. Changing any of them requires a new
  decision record here, per the `CLAUDE.md` rule on grid ratio, fields, split seeds,
  loss, and evaluation protocol.

  | Item | Frozen value |
  |---|---|
  | Channel order | `[eta, u, v]`, always; `h = H + eta` reconstructed, never stored |
  | Grid layout | raw: staggered C-grid, `[x, y]`; stored: `[time, channel, y, x]`; processed: cell-centered (D011) |
  | Batch contract | `[B, 3, H, W] -> [B, 3, 4H, 4W]`, resolution-generic; required smoke shapes H=W=32 and H=W=64 |
  | Interpolation | bicubic, `align_corners=True`, shared by the M-01 baseline and both models' residual path |
  | Residual form | `y_hat = bicubic(x) + R_theta(x)` for both models |
  | Normalization | per-channel, fit on fine-grid **train** states only, per pair ID; `q' = (q - mu) / max(sigma, 1e-8)` |
  | Loss | equal-weight normalized channel MSE, macro-averaged over the three channels |
  | Selection metric | lowest full-validation normalized MSE |
  | Pair IDs | `swe_gaussian_32x128_v1` (primary), `swe_gaussian_64x256_v1` (backup) |
  | Trajectory IDs | stable UUIDs in `ic_registry_v1.json`; splits by seed ID 0-31 train, 32-39 val, 40-47 test |
  | Fresh workload IDs | `fresh_id`, `ring_ood`; evaluation-only, absent from any training manifest |
  | Primary seed | `20260812` |
  | Metrics | per-channel normalized MSE, physical RMSE (SI), relative L2, max abs error; aggregated per snapshot, then within trajectory, then equal-weight across trajectories |

- Reason: `docs/AGENT_WORKFLOW.md` requires PDE/Data and ML to work in parallel against
  a frozen contract, with ML using synthetic fixtures until real data pass G3. This is
  that contract in one place.

## D017 - Trajectories run to the reference solver's duration, 197 snapshots per pair

- Status: accepted, supersedes the 128-snapshot figure in D007
- Decision: integrate each trajectory for the same number of updates the reference
  `swe.py` performs, 4999 on the primary pair's time step, and save 197 snapshots. The
  primary discards 288 steps and saves every 24 to step 4992; the backup discards 576 and
  saves every 48 to step 9984. Both pairs keep their existing strides.
- Reason: two findings from measuring the data rather than assuming.

  First, temporal density was not the limiting factor. Field autocorrelation against step
  lag on a real trajectory gives 0.9999 at lag 1, 0.9690 at lag 24, and a zero crossing at
  roughly lag 150. So a trajectory holds only about 22 statistically independent states at
  the old 3336-step length however densely it is sampled, and the previous 128 snapshots
  already oversampled that by roughly six times. Saving every step would have cost 26
  times the bytes for frames 99.99 percent correlated with their neighbours.

  Second, what does help is a longer trajectory. Extending to 4992 steps raises the
  independent-state count per trajectory from about 22 to about 33, a genuine 50 percent
  gain in independent training signal, for 1.54 times the storage.
- Consequences:
  - Snapshot pairs per pair ID rise from 6,144 to 9,456: 6,304 train, 1,576 validation,
    1,576 test. Any report quoting 1,024 held-out test snapshots is stale.
  - Primary raw payload rises from 1.195 GiB to 1.840 GiB, still inside the sub-2-GiB raw
    target in `docs/PROJECT_SPEC.md`, so D007's storage budget stands. The backup rises
    from 4.781 GiB to 7.359 GiB and the combined payload to 9.198 GiB.
  - Generation cost stays small: about 1.7 min for the full primary release and 10.8 min
    for the backup, measured at 1.44 s per paired primary trajectory.
  - The backup's stride and step cap are both double the primary's, not because of the
    snapshot count but because its `dt` is half: its fine grid has half the spacing, so
    twice the steps cover the same physical time. Doubling both keeps the pairs aligned in
    physical duration (34.86 h against 34.72 h, the same 0.392 percent mismatch already
    recorded) and in frame count (197 each). Capping both at the same *step* count would
    instead have left the backup covering half the primary's evolution, which would
    invalidate the paired cross-resolution bootstrapping this file's D008 relies on.
  - Verified admissible at the longer durations before adopting: worst relative mass drift
    1.65e-15, minimum total depth 98.500 m, gravity CFL unchanged at 0.100 on both fine
    grids, no non-finite values, across all four grids.
  - The training-trajectory-count ablation O-02 remains the right way to test whether
    trajectory count matters more than snapshot count; this decision does not pre-empt it.

## Template for new decisions

```text
## DNNN - Title

- Status: proposed | accepted | superseded
- Decision:
- Reason:
- Consequences:
```
