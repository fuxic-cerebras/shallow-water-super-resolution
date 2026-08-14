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

## D018 - Augmentation implemented but disabled by default

- Status: accepted, corrects `docs/DATASET.md`
- Decision: implement the reflection and transpose transforms with correct vector handling
  and keep them available behind config, but apply **no** augmentation by default.
- Reason: `docs/DATASET.md` calls reflection and transpose "symmetry-preserving". For this
  solver they are not. Measured discrepancy between transforming-then-evolving and
  evolving-then-transforming, as a fraction of the evolved field's peak amplitude:

  | Configuration | `reflect_x` | `reflect_y` | `transpose` |
  |---|---:|---:|---:|
  | Coriolis + beta (project default) | 0.914 | 0.919 | 0.934 |
  | Coriolis, no beta | 0.849 | 0.908 | 0.862 |
  | No Coriolis | 0.048 | 0.048 | **0.000** |

  Two independent causes, which the no-Coriolis row isolates:

  1. **Rotation is chiral.** A mirror image of a rotating system rotates the other way, so a
     reflection is a symmetry only if `f -> -f`. Here `f` runs from 9e-5 to 1.1e-4 and is
     positive everywhere, so it never is. The transpose additionally maps `f = f_0 + beta*y`
     onto an `x` dependence. This dominates, at roughly 0.9.
  2. **The C-grid staggering is not reflection-symmetric.** `u_i` sits at `x_{i+1/2}`, offset
     east, so reflecting maps an east-face variable onto a west-face position, a different
     index alignment. Worth 0.048 on its own, and it is why the reflections stay broken even
     with rotation disabled, while the transpose becomes exact.
- Consequences:
  - The transforms are geometrically exact on the destaggered, cell-centered fields the
    loader produces: the grid is uniform and square, so flipping or swapping axes maps grid
    points onto grid points. What they are not is *dynamically* valid -- an augmented pair
    represents a state this solver could not produce.
  - Enabling them is therefore an empirical question, not a free win: the model could learn a
    reflection equivariance that the true coarse-to-fine operator does not have, because the
    coarse solver's error structure is chirally biased by rotation. `docs/EXPERIMENT_PLAN.md`
    says not to add complexity before the MSE-only baseline exists, so the default is off and
    an ablation can measure it later.
  - `augmentation_symmetry_error` in `swe_sr/data/processing.py` keeps this measurable rather
    than a comment, and the tests assert the failure as a lower bound: if a future change made
    these symmetries, that would be a significant physics change and should fail loudly.
  - The transpose's gate in `docs/DATASET.md` ("use it only after a unit test verifies the
    transformation") is satisfied by `tests/data/test_processing.py`, which verifies both the
    axis swap and the `u`/`v` channel swap, and includes the wrong-vector-reflection negative
    test from `docs/VALIDATION.md`.

## D019 - Normalization is fitted on destaggered fine training fields

- Status: accepted
- Decision: fit per-channel statistics on the **destaggered** fine-grid training fields, not
  the raw staggered arrays, and record counts, sums, and sums of squares alongside the mean
  and standard deviation.
- Reason: models consume the processed cell-centered representation (D011), and destaggering
  is an averaging operator that reduces velocity variance. Fitting on raw arrays would leave
  the model's inputs systematically mis-scaled. `docs/DATASET.md` says "fine-grid training
  states" without specifying which layer, so this pins it.
- Consequences: statistics are accumulated in float64 by streaming one trajectory at a time,
  so memory stays flat over a multi-GiB release. The three accumulators let an auditor
  re-derive the statistics independently instead of trusting the generator. The generator now
  also writes the `processed/<dataset_id>/{manifest.json,normalization.json}` layer that
  `docs/DATASET.md` specifies and that `CLAUDE.md`'s `validate` command points at.

## D020 - T-03 experiment freeze

- Status: accepted
- Decision: the two completed 30,000-step runs recorded in `docs/EXPERIMENT_FREEZE.md` are the
  T-03 primary experiment. That file pins the dataset, manifest hash, IC registry hash, launch
  commits, config hashes, seed, normalization statistics, metric definitions, checkpoint rule,
  checkpoint SHA-256 digests, and the held-out test results. Authorized by the project owner on
  2026-08-13.
- Reason: the diagnostic runs answered the question they were launched for and the evidence is
  sufficient to freeze. Both models trained the full 30,000 steps on the frozen manifest with
  the primary seed, both were independently recomputed from arrays, and the outstanding
  lead-time concern resolved: U-Net matches bicubic at the shortest lead time (0.0103 against
  0.0090) rather than losing badly, so no design change is required and stratified reporting is
  sufficient.
- Consequences:
  - The runs keep `stage: diagnostic` inside their own artifacts, because that is what they were
    launched as. Rewriting a completed run's provenance to match a later decision would falsify
    it. The freeze record is the designation; `swe_sr.report` reads that file so a frozen run is
    identified correctly without its artifacts being altered.
  - The two runs were launched from different commits, `077d6b53` and `da865691`. This was
    verified immaterial rather than assumed: the only training-path difference is the addition of
    `swe_sr/data/fresh.py`, which training never imports. `train.py`, both models, the data
    pipeline, the loss metric, and all configs are byte-identical across that range.
  - A provenance flaw was found and fixed while preparing this freeze. `summary.json` recorded
    `git_commit` as HEAD when the summary was written, not when the code was loaded, so a run
    during which anyone commits would be attributed to code it never executed. `train.py` now
    records `git_commit` at launch and `git_commit_at_completion` separately. The two frozen runs
    predate the fix, so their accurate launch commits are taken from their `run_id` values and
    stated explicitly in the freeze record.
  - Superseding this freeze requires a new decision and a new freeze file, not an edit to the
    existing one.

## D021 - Cross-resolution transfer is an evaluation, and never writes the canonical artifact

- Status: accepted
- Decision: The 64->256 release (`swe_gaussian_64x256_v1`) is generated in full and used **only**
  to evaluate the frozen 32->128 checkpoints. No model is trained on it, and no checkpoint is
  selected or tuned against it. A cross-pair evaluation writes
  `evaluation_{split}__{pair_id}.json`; the canonical `evaluation_{split}.json` is reserved for
  the pair a checkpoint was trained on.
- Reason: The owner asked for a transfer test rather than a second training run, so the 64->256
  data is evaluation-only by construction and D008's rule that the two pairs never mix in one
  training run is untouched. The filename rule exists because `evaluate.py` previously derived
  the artifact path from the split alone: running the documented `--manifest` override against a
  frozen run would have replaced the frozen T-03 `evaluation_test.json` -- the file
  `docs/RESULTS.md`, `docs/EXPERIMENT_FREEZE.md` and `scripts/verify_independent.py` all read --
  with numbers from a different dataset, silently. Deriving the name from the evaluated pair
  makes that impossible rather than relying on a flag the caller must remember.
- Consequences:
  - The full 48-trajectory 64->256 pair is generated (5.3 GiB) because normalization must be
    fitted on its own train split (D019); generating only the 8 test trajectories would have
    forced either a cross-pair statistics violation or a statistic fitted on evaluation data.
    Generating it does **not** discharge D-06 or O-01, which remain deferred: this release exists
    for evaluation and no training run has used it.
  - Both pairs' statistics were compared and are within 0.5% on every channel, so the transfer
    test isolates spatial scale rather than amplitude. A frozen-statistics variant was judged
    non-separable at that margin and not run.
  - `trajectory_means_macro_mse_normalized` is now serialized in every evaluation artifact. Two
    models evaluated in separate processes could otherwise only be compared by asking whether
    their independent confidence intervals overlap, which is weaker than the paired test the
    protocol already prescribes. This is an additive schema change; the frozen in-distribution
    artifacts are **not** regenerated to acquire the field, because re-running them would also
    rewrite host-load-dependent `seconds_per_frame` inside a frozen record.
  - Findings are recorded in `docs/experiments/O-07-cross-resolution-transfer.md`, including one that revises how earlier results
    should be read: both models lose to bicubic at short lead time on the trained pair too, which
    the aggregate concealed.

## D022 - Direct-prediction ablation arm

- Status: accepted
- Decision: add `outer_baseline: bicubic | none` to both model configs and run ablation 3 of
  `docs/EXPERIMENT_PLAN.md` ("outer bicubic residual versus direct prediction") for both
  architectures on the frozen manifest, seed, and schedule. Authorized by the project owner on
  2026-08-13. The frozen T-03 runs and `docs/RESULTS.md` are **not** touched; the comparison is
  written to `docs/experiments/A-01-residual-vs-direct.md`.
- Reason: D006 fixed the outer residual for two reasons, and only one of them was ever
  demonstrated. The comparison-fairness reason is a proof: zeroed weights reproduce bicubic
  exactly, so any gain over bicubic is attributable to the learned residual. The optimization
  reason was never tested against a direct-prediction control. The owner's specific hypothesis
  is that for EDSR the additive bicubic path may impose a bias the network then has to fight,
  which is consistent with the `r`/`c` decomposition in `docs/experiments/O-07-cross-resolution-transfer.md` measuring correction
  magnitudes up to 9.5x too large at short lead time — exactly where an unconditionally added
  baseline does the most damage.
- Consequences:
  - The two arms have **identical parameter counts** (1,930,208 and 1,517,571), so the ablation
    varies exactly one factor, as `docs/EXPERIMENT_PLAN.md` requires of an ablation.
  - What "direct" means differs by architecture, and this is stated rather than smoothed over.
    For EDSR it is a return to the published form with no interpolation anywhere. For the U-Net
    the encoder still receives the bicubic-upsampled input, because operating on the output grid
    is structural to it; only the additive skip is removed. Both isolate the outer additive
    baseline.
  - The D006 guarantee that a zero-weight model reproduces bicubic does **not** hold for the
    direct arms, and `tests/models/test_models.py` asserts that it does not. That is the negative
    control proving the flag changes the graph.
  - `model_config_for_run` now resolves the architecture from a run's recorded `config.yaml`
    rather than from `configs/model/<model_name>_x4.yaml`. The old rule was unsafe once two arms
    shared parameter shapes: `load_state_dict` against the wrong arm would succeed silently and
    every reported metric would describe a model that was never trained. All seven call sites in
    `swe_sr.evaluate`, `swe_sr.evaluate_fresh`, and the four scripts now use it, with a fallback
    to the old convention for runs predating the field, and a regression test.
  - Model names are `unet_direct` and `edsr_direct`, which flow into run IDs, so an ablation run
    can never be mistaken for a frozen T-03 run in any artifact or figure.

## D023 - ConvMixer as a third architecture

- Status: accepted
- Decision: add ConvMixer (Trockman & Kolter, arXiv:2201.09792) as a third trained
  architecture, `ConvMixer-256/16` with `k=9, p=1`, on the frozen manifest, seed, and schedule.
  Authorized by the project owner on 2026-08-13. The frozen T-03 U-Net and EDSR runs are **not**
  retrained; the third arm is added to `docs/RESULTS.md` alongside them.
- Reason: the two existing models span a narrow slice of design space, and the axis they leave
  untested is the one this problem is most likely to care about. The U-Net reaches basin-scale
  context by pooling down a pyramid. EDSR never downsamples, but 16 stacked 3 x 3 convolutions
  see only about 33 px — barely one grid width of a 32 x 32 input. Neither isolates the
  question of whether a **global receptive field without any pooling** helps in a closed
  rotating basin where gravity waves reflect off the walls. ConvMixer is the minimal probe for
  that: it is isotropic, holding one resolution end to end, and buys range purely through
  unusually large depthwise kernels. At the chosen depth its receptive field is 129 px at low
  resolution, wider than both the 32 x 32 and 64 x 64 grids, so one unit integrates the whole
  domain with no pyramid at all. It is also about six lines of PyTorch, so the arm is cheap to
  add and hard to get subtly wrong.
- Consequences:
  - **1,720,067 parameters**, between EDSR's 1,517,571 and the U-Net's 1,930,208. The comparison
    stays about inductive bias rather than capacity, and
    `test_the_three_architectures_have_comparable_capacity` pins both the exact count and the
    ordering.
  - Hyperparameters are taken from the paper's **CIFAR-10** ablations rather than its ImageNet
    ones, because those are run at 32 x 32 — exactly this project's low resolution. `k=9` is the
    knee of the kernel sweep (3 -> 93.61%, 5 -> 95.11%, 7 -> 95.72%, 9 -> 95.88%, +0.28% beyond),
    and the paper separately shows large kernels beat equal-parameter extra depth. `p=1` is the
    same sweep's preference at that input size (p=2 costs 0.80%, p=4 costs 3.27%); patching also
    discards precisely the high-frequency content super-resolution exists to recover, so the stem
    is a pointwise lift and all upsampling is deferred to the decoder. `patch_size` remains a
    config field so p>1 is a one-line follow-up ablation.
  - **BatchNorm is retained as published, and this contradicts EDSR.** EDSR's central claim is
    that batch normalization degrades super-resolution, and `swe_sr/models/edsr.py` and
    `unet.py` both omit normalization for that reason. ConvMixer's own ablation measures 1.44%
    on CIFAR-10 for BatchNorm over LayerNorm, so the layer is doing real work in this
    architecture. Rather than resolve the disagreement by assumption, the arm keeps the
    published block and lets the benchmark measure it. If ConvMixer underperforms, BatchNorm is
    a live confound and the honest follow-up is a normalization ablation, not a retroactive
    reinterpretation.
  - The cost of that choice is bounded and tested. A training-mode forward pass genuinely
    couples samples within a batch. Every path in this repository that reports a number calls
    `model.eval()` first, which switches BatchNorm to fixed running statistics and restores
    exact per-sample independence, so no reported metric depends on batch composition.
    `test_convmixer_batch_coupling_is_confined_to_training_mode` asserts both halves — that the
    coupling is real while training, and that it is gone under `eval()` — so neither can drift.
  - `test_no_batch_or_instance_normalization_anywhere` was renamed to
    `test_no_normalization_in_the_two_reference_architectures`. Its assertion is unchanged and
    was never weakened: it only ever iterated `build_unet()` and `build_edsr()`, so the old name
    overstated its scope. `test_convmixer_keeps_batchnorm_as_published` is the positive
    counterpart, so ConvMixer's exclusion cannot quietly become an accident.
  - The D006 guarantee still holds for this arm: zeroed weights reproduce bicubic **exactly**.
    BatchNorm does not break it, because zeroing `gamma` and `beta` makes every normalization
    emit exact zero, and the existing bitwise test covers ConvMixer through the shared fixture.
  - Checkpoints now carry buffers that matter. `state_dict` includes BatchNorm running
    statistics and `torch.save`/`load_state_dict` round-trip them, which the bitwise reload test
    covers. One consequence to watch: the cross-resolution transfer evaluation applies running
    statistics estimated at 32 x 32 to a 64 x 64 input. Inputs are identically normalized so the
    statistics should transport, but if ConvMixer's transfer degrades markedly more than the
    other two models', that is a finding about BatchNorm and belongs in `docs/experiments/O-07-cross-resolution-transfer.md`
    rather than being treated as a defect.
  - Two departures from the paper are structural to the task and not optional. The classifier
    head (global average pooling plus a linear layer) would destroy the spatial field, so it is
    replaced by EDSR's decoder. A 1 x 1 projection to 64 channels precedes that decoder because
    `PixelShuffleUpsampler(features)` costs `36h^2 + 4h`; at h=256 a full-width decoder would be
    4.7 M parameters, larger than the entire body. Projecting first makes the decoder
    byte-identical to EDSR's, so the two architectures differ in their body and not their
    decoder.
  - The model name is `convmixer` — lowercase and underscore-free — which the run-ID regex in
    `swe_sr/report.py` and the config-path fallback in `swe_sr/training/config.py` both require.
  - The D022 direct-prediction arm is **not** shipped for ConvMixer. `outer_baseline="none"`
    works via the base class, but no config or run exists for it; it is recorded as a follow-up
    in `TASKS.md` rather than silently implied.
- Pilot evidence (2026-08-13, Slurm job 301044, node `cpu-dy-x48-m7a-2`, 16 threads, BF16),
  which revises the risk assessment above and is recorded because it was surprising:

  | arch | pilot train | pilot val | gap |
  |---|---:|---:|---:|
  | EDSR | 0.1726 | 0.3069 | 1.8x |
  | U-Net | 0.1711 | 0.3028 | 1.8x |
  | ConvMixer | 0.0713 | 0.5786 | 8.1x |

  ConvMixer fits the training data 2.4x better than EDSR and generalizes 1.7x worse. **The
  cause is not BatchNorm.** Scoring one checkpoint on one validation split with only the
  normalization mode varying gives 0.5287 under running statistics against 0.4334 under batch
  statistics — a factor of 1.22, not 8 — and the running statistics are well conditioned
  (`running_var` spanning 9.3e-4 to 0.97 across all 33 layers, 1,823 batches tracked). The
  BatchNorm concern recorded above is therefore real but second order, and the eval-mode
  contract is doing its job. `scripts/diagnose_convmixer_bn.py` reproduces this.

  The likely mechanism is regularization, and it is in the paper's own Table 3: its two largest
  effects are not architectural but augmentation — removing RandAugment costs 2.96% and removing
  random scaling 9.64%, each larger than kernel size, patch size, or the normalization choice.
  ConvMixer is a high-capacity model the paper controls with heavy augmentation. This project
  runs with augmentation off (D018, because reflections are not symmetries of a rotating
  beta-plane) and weight decay at 1e-6, so the arm operates in exactly the regime the paper
  identifies as most damaging to it. That is a property of the comparison, not a defect, and it
  must be stated when the arm is reported: this benchmark tests ConvMixer *unregularized*.

  The full run proceeds on the identical schedule regardless, because changing weight decay or
  augmentation for one architecture alone would break the single-factor discipline the
  comparison depends on. The pilot trains on 8 of 32 trajectories and pilot numbers do not
  predict full ones (EDSR went 0.3069 pilot to 0.0830 full test), so the full run is the
  measurement of record. `TASKS.md` A-05 carries the regularization ablation as the honest
  follow-up.

  The full run settled the question: at epoch 8 of 39 ConvMixer reached validation 0.0653 with
  a train/validation gap of 1.79x, indistinguishable from EDSR's and the U-Net's 1.8x. The
  pilot's 8.1x was an artifact of its 8-trajectory subset and nothing more.
- **A-03, the unnormalized arm** (`configs/model/convmixer_nonorm_x4.yaml`, owner-requested
  2026-08-13). Read it as "ConvMixer as EDSR would have designed it" versus "ConvMixer as
  published", **not** as a single-factor BatchNorm ablation. It cannot be the latter, and that
  is itself the first result: BatchNorm cannot simply be deleted from this architecture,
  because the pointwise stage is not residual and 16 unnormalized non-residual layers compound.
  Measured first-block depthwise gradient norm, against the last block:

  | variant | first/last gradient | activation std | trainable |
  |---|---:|---:|---|
  | BatchNorm, as published | 13 | 1.000 | yes |
  | normalization removed, nothing else | 3.9e-07 | 0.021 | no, gradient dies |
  | + residual around the pointwise | 0.81 | 49.5 | no, activations diverge |
  | + residual and `res_scale=0.1` | 0.90 | 0.409 | yes |

  So the arm changes three values together — `normalization: none`, `pointwise_residual: true`,
  `res_scale: 0.1` — which is precisely EDSR's recipe of residual-everywhere, scaled,
  unnormalized. Running the naive one-factor deletion would have measured only that an
  unnormalized non-residual stack does not train, which is already known.
  `test_removing_normalization_alone_would_not_train` pins the measurement so that a future
  "simplification" to a clean one-factor ablation fails loudly rather than producing a dead run.
  Capacity is 1,703,171 against 1,720,067, a 0.98% difference that is unavoidable because
  removing BatchNorm removes its affine parameters; the residual and the scaling add none.
  The arm gains one property the published one lacks: with no batch statistics anywhere it is
  batch-independent in training mode too, not only under `eval()`.

## D024 - Documented numbers are generated, and documents are organized by lifetime

- Status: accepted
- Decision: no measured number is written into prose by hand. Every run the documentation cites
  is registered in `docs/results/runs.yaml`; `python -m swe_sr.results --write` builds
  `docs/results/index.json` from those runs' artifacts; `python -m swe_sr.docgen render` renders
  the documents' tables from that index into marker-delimited blocks; and
  `python -m swe_sr.docgen check` runs in `scripts/check.sh`, so a stale table fails the build.
  Alongside it, documents are organized by how they change over time rather than by topic, per
  `docs/DOCUMENTATION.md`.
- Reason: the documentation was growing faster than the work, and the cost was not page count
  but duplication. Measured against the tree before this decision, the headline `0.0400`
  appeared in eight files and `0.0830` in nine, each transcribed by hand, so re-running an arm
  required editing every one of them consistently with nothing to detect a missed edit. The
  drift had already started: `README.md` claimed the decision log ran "D001 to D021" when D023
  existed, and its documentation index omitted two write-ups that its own prose linked. `runs/`
  is gitignored, so neither CI nor a reader could re-derive any of it — which is why the index
  is committed rather than generated on demand.
- Consequences:
  - **A found defect, which is the argument for the whole change.** Converting the A-05 and A-03
    paired tables to generated blocks showed their confidence intervals were paired
    t-intervals, `mean +/- t_{7,0.975} * SE`, not the percentile bootstrap over trajectories
    that `docs/VALIDATION.md` mandates and that every other paired interval in the project uses.
    On these eight pairs the t-interval is symmetric by construction and 10-20% wider. Every
    mean difference, verdict, and win count is unchanged, so no conclusion moves, but two
    documents had been stating an interval computed by an undocumented estimator. Nothing in the
    repository could have caught that while the numbers were typed.
  - Three kinds of document, and the kind determines what may grow. **Contracts**
    (`docs/PROJECT_SPEC.md`, `docs/DATASET.md`, `docs/ARCHITECTURE.md`,
    `docs/EXPERIMENT_PLAN.md`, `docs/VALIDATION.md`) are normative, contain no results, and stay
    short. **Ledgers** (this file, `TASKS.md`, `docs/EXPERIMENT_FREEZE.md`) are append-only and
    are *expected* to grow without bound, provided each entry is bounded and an index sits at
    the top. **Experiment write-ups** (`docs/experiments/`) are immutable once their arm lands:
    a new result is a new file, never an edit to an existing write-up.
  - Two kinds of block. A `generated` block is rewritten by `render`. A `verified` block is
    checked and **never** rewritten, which is what `docs/EXPERIMENT_FREEZE.md`'s results table
    is: silently rewriting a freeze to agree with whatever the artifacts now say would destroy
    the only property a freeze provides. A mismatch there demands a new decision and a new
    freeze, not an edit.
  - `docs/results/index.json` is committed although it is generated. It is not experiment data
    in the sense `.gitignore` excludes — no arrays, no checkpoints, no logs, only the ~150 KB of
    scalars the documents cite — and committing it is what lets CI verify a table in a clone
    with no `runs/`. The same reasoning as the IC registry exception.
  - `seconds_per_frame` is excluded from the index. It is wall-clock time under unknown host
    load, it is not comparable across arms, and it has already misled once, so it does not
    belong in a file whose purpose is to be cited. Like-for-like timing comes from
    `scripts/time_inference.py` or from a single run's `metrics.csv`.
  - `swe_sr/report.py` no longer regexes run IDs out of `docs/EXPERIMENT_FREEZE.md` to decide
    which runs are frozen; it reads the registry. The old pattern made a prose file load
    bearing, and it could not match a model name containing an underscore.
  - Four hygiene rules now fail CI: a backticked repository path that does not resolve, a
    document no other document references, a `DNNN` citation with no entry in this file, and a
    run ID cited in prose but absent from the registry. The path rule deliberately skips paths
    rooted outside this repository, because `docs/RESEARCH_MATRIX.md` correctly cites upstream
    files that do not exist here.
  - `docs/RESULTS.md` remains a wholly generated *file* rather than a set of blocks, and is not
    covered by `docgen check`: it reports `ms/frame`, so regenerating it on a differently loaded
    host produces a diff that means nothing. Numbers cited elsewhere come from the index.

## Template for new decisions

```text
## DNNN - Title

- Status: proposed | accepted | superseded
- Decision:
- Reason:
- Consequences:
```
