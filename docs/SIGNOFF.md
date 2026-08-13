# I-03 — Integration and Science Sign-Off

Audited 2026-08-13 against the frozen T-03 experiment (`docs/EXPERIMENT_FREEZE.md`, D020).

**Verdict: signed off with two remaining exceptions** (a third was cleared on 2026-08-13). Every claim below was checked rather than
assumed, and where a check could not be completed that is recorded as an exception rather than
waived. `docs/AGENT_WORKFLOW.md` forbids waiving a failed gate without a decision record; no gate
failed, and the exceptions are scope and environment limits, not failures.

## 1. Definition of done (`docs/PROJECT_SPEC.md`)

| Requirement | Status | Evidence |
|---|---|---|
| Raw and processed manifests reproduce all 48 paired trajectories | **pass** | `python -m swe_sr.data.validate` passes 12/12 gates; 384 array checksums recomputed |
| Split auditing proves no trajectory appears in more than one split | **pass** | validator gate plus independent recomputation from manifest rows; 48 unique IDs, 0 overlaps |
| U-Net and EDSR each produce a best-validation checkpoint and curves | **pass** | both runs: `checkpoints/best.pt`, `metrics.csv`, `curves.png`; best epochs 39 and 34 |
| Both checkpoints evaluated on all 1,576 held-out test snapshots | **pass** | `evaluation_test.json` in each run, 1,576 snapshots over 8 trajectories |
| Both checkpoints evaluated on a workload generated after training | **pass** | `fresh_id` and `ring_ood` for both frozen checkpoints, generated post-selection |
| Nearest and bicubic use the identical test states and metrics | **pass** | asserted in `tests/test_evaluate.py`; all methods report identical snapshot counts |
| A machine-readable metrics file and human-readable report exist | **pass** | `evaluation_*.json` and `docs/RESULTS.md` |
| Commands and environment metadata suffice to reproduce every result | **pass** (exception 3 notes one limit) | fresh clone plus fresh venv reproduces the gate and the identical IC registry hash |

## 2. Research questions, answered from measured results

1. **Can either network beat bicubic?** Yes. U-Net 0.0400 and EDSR 0.0830 against bicubic 0.4295
   normalized macro MSE; paired bootstrap intervals `[-0.4920, -0.2805]` and `[-0.4334, -0.2514]`
   both exclude zero.
2. **Does U-Net outperform EDSR?** Yes in distribution, by 2.1x, for 1.27x parameters and about
   1.9x compute. Note a 2,000-step pilot showed them tied at 0.303 and 0.307, so this required
   the full 30,000 steps to establish.

   *Corrected 2026-08-13.* This read "2.2x inference cost", from the `ms/frame` column of two
   separate evaluations. That column is not a controlled cross-run comparison: the runs were
   evaluated on a shared host at different times, and the bicubic control — identical work in
   both — reads 0.2 against 0.3 ms/frame, so the U-Net evaluation's host was measurably more
   loaded. The ratio now quoted is within-run training throughput over 30,000 steps on a fixed
   16-thread allocation, 79.7 against 42.0 samples/s median. The accuracy conclusion is
   unaffected; only the cost ratio was contaminated.
3. **Are gains consistent across channels?** Yes. U-Net relL2 is 0.1951 / 0.2149 / 0.2135 for
   `eta` / `u` / `v`; the velocity channels are within about 10% of elevation, so no channel is
   carried by the others.
4. **Does lower pixel MSE also improve physical diagnostics?** **No.** U-Net improves relative
   mass error over bicubic (0.0347 against 0.0392) but EDSR degrades it to 0.0540 while beating
   bicubic fivefold on MSE. Pixel accuracy and physical consistency come apart, and this is the
   clearest negative result in the study.
5. **How sharply does performance degrade on a fresh ring-wave workload?** Sharply, and the
   ranking inverts. On `ring_ood`, EDSR 0.3246 against U-Net 0.3505 — U-Net's in-distribution
   advantage does not transfer. On `fresh_id` both are slightly *better* than on the test split
   (0.0303 and 0.0666), so the degradation is specific to the unseen initial-condition family
   rather than to fresh data as such.

## 3. Verification independent of the code under test

`scripts/verify_independent.py` reimplements normalization, destaggering, every field metric, the
mass diagnostic, and the aggregation protocol in plain numpy from the specification, importing
only the HDF5 reader and the model definition. Both frozen runs: **20 checks each, all passing**,
covering normalization re-derived from raw accumulators, split disjointness, sampled array
checksums over canonical bytes, macro MSE, all three relL2 channels, and mass error.

That design earned itself. It found an evaluation metric whose value depended on batch size —
relL2 and mass error were being averaged per batch rather than reduced by the documented
protocol, off by 7% on the real bicubic case. Reusing the package's own metrics could not have
caught it.

## 4. Findings that revise the specification

Recorded as decisions rather than left implicit, because each contradicts something the original
specification assumed:

- **D011** — the solver is an Arakawa C-grid, so storing `[eta, u, v]` as colocated channels is
  off by half a cell for velocities, and the offset differs by resolution. Velocities are
  destaggered in the processed layer.
- **D014** — `docs/VALIDATION.md` deferred to a discrete energy convention that no document
  defined. One is now defined.
- **D015** — no CUDA device is reachable; the runtime budget is the Slurm CPU partition in BF16.
- **D017** — trajectories extended to the reference solver's own duration, 197 snapshots.
- **D018** — the "symmetry-preserving" augmentation the specification prescribes is **not**
  symmetry-preserving for a rotating beta-plane; measured discrepancy about 0.9 relative. It is
  implemented but disabled by default.
- **D019** — normalization is fitted on destaggered fields, since that is what models consume.

## 5. Exceptions, stated rather than waived

**Exception 1 — CLEARED 2026-08-13.** The frozen release now sits at the canonical
`data/{raw,processed}/swe_gaussian_32x128_v1/`, so the documented commands work as written.
Re-verified after the move: the processed manifest hash is unchanged at `af02e44f...`, validation
passes 12/12 at the canonical path, and both frozen runs re-evaluate to identical numbers. The
superseded development dataset was moved aside rather than deleted, and `data/staging/` now holds
symlinks so the frozen runs' recorded manifest paths stay resolvable without their artifacts being
edited. See the promotion section of `docs/EXPERIMENT_FREEZE.md`.

**Exception 2 — the backup 64->256 pair is deferred.** D-06 and O-01 are outside this sign-off by
owner decision. Their tests are marked `backup` and deselected, not deleted, so coverage returns
with `pytest -m backup`. The backup design, configs, and decisions D008 and D017 stand.

**Exception 3 — clean-install reproduction did not resolve the torch wheel.** The reproduction
used a fresh clone with `--recurse-submodules`, a fresh venv, and `pip install -e ".[dev]"`,
verified with the gate passing both with and without matplotlib installed. Packaging, imports,
entry points, and the documented `generate` and `validate` commands were exercised fresh, and
generation reproduced the identical IC registry hash. Only torch wheel resolution was not
re-exercised locally; CI covers it. An earlier attempt used `--system-site-packages`, which
masked a missing optional dependency and let a CI failure through — that is why this exception is
stated specifically rather than as a general caveat.

## 6. Sign-off status and its limits

`TASKS.md` marks tasks `ready-for-review`, not `complete`, with one exception: I-02, which the
project owner authorized directly. That is deliberate. `CLAUDE.md` and `docs/AGENT_WORKFLOW.md`
require that authors stop at `ready-for-review` and that only an independent verifier and the
integration lead record completion. A single agent performed every role in this project, so the
independence those gates assume was not available, and marking work `complete` would assert a
separation that did not exist.

What was done instead: the verifier reimplements from the specification rather than reusing the
code it checks, negative tests establish that each gate fires rather than merely passes, and
every corrected claim is recorded in the history rather than quietly amended. That is weaker than
genuine independent review and should not be mistaken for it.

**Recommended before external release:** a review by someone other than the author of the
following, which are the places where a single-agent process is least trustworthy — the
destaggering decision D011 and its effect on reported velocity errors; the aggregation protocol
implementation, given a batch-dependence defect was already found there; and the D018 claim that
the prescribed augmentation is invalid, since it overturns the specification.
