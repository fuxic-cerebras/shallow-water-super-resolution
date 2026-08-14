# Frozen Experiment Record — T-03

Authorized by the project owner on 2026-08-13. This file **is** the freeze: it designates the
two runs below as the T-03 primary experiment and pins every identifier a reader needs to
reproduce or audit them. Per `docs/AGENT_WORKFLOW.md` gate G6, the manifest, commit, configs,
statistics, seed, metrics, and checkpoint rule are frozen. Changing any of them requires a new
decision record and a new freeze, not an edit to this one.

## An honesty note on the stage label

Both runs carry `stage: diagnostic` inside their own `config.yaml` and `summary.json`, because
that is what they were launched as — they were run to test whether an early-lead-time regression
was an undertraining artifact. Those artifacts are **not** rewritten to say `full`. Editing a
completed run's provenance to match a later decision would be falsifying it. The freeze is
recorded here instead, and `swe_sr.report` reads this file so a frozen run is identified
correctly without its artifacts being altered.

## Dataset

| Item | Value |
|---|---|
| `dataset_id` | `swe_gaussian_32x128_v1` |
| `pair_id` | `swe_gaussian_32x128_v1` |
| IC registry hash | `976e3a577a25a633c6a2625263f23e60482768965029805a5efd16be97ab7c8c` |
| Processed manifest hash | `af02e44fbc764212f1e011c9db9b20e9eda61fd1c4da552b5545516d70cf4825` |
| Generated at commit | `c82ce16ac0e5dd3d9d0393d100947a7d222df955` (clean tree) |
| Split | 32 train / 8 validation / 8 test trajectories, by seed ID (D004) |
| Snapshots | 197 per trajectory, last step 4992, 34.86 h physical (D017) |
| Shared time step | 25.139798 s (D003) |
| Location | `data/{raw,processed}/swe_gaussian_32x128_v1/` (canonical, promoted 2026-08-13) |

### Frozen normalization (train split only, fine grid, destaggered; D019)

| Channel | mean | std | count |
|---|---|---|---|
| `eta` | `+1.758229280e-02` | `1.761697024e-01` | 103,284,736 |
| `u` | `+7.715977525e-05` | `3.945340279e-02` | 103,284,736 |
| `v` | `-2.582595058e-04` | `3.975597263e-02` | 103,284,736 |

## Runs

| Item | EDSR | U-Net |
|---|---|---|
| `run_id` | `20260812T230157Z_edsr_aae64836_077d6b53` | `20260812T235727Z_unet_e3ce47d7_da865691` |
| Model config | `configs/model/edsr_x4.yaml` | `configs/model/unet_x4.yaml` |
| Experiment config | `configs/experiment/full_diagnostic.yaml` | same |
| Config hash | `aae648363d26465cef444a8fcbe00315d038b9f3b35033aa15eb53274d4668e5` | `e3ce47d776b12c54753df94d58054c50dc6e2bad23c9911bb59093b8aae42de2` |
| Code commit at launch | `077d6b53` | `da865691` |
| Seed | 20260812 | 20260812 |
| Precision | BF16 (AVX512-BF16) | BF16 |
| Optimizer steps | 30,000 (cap reached) | 30,000 (cap reached) |
| Best epoch | 34 | 39 |
| Best validation MSE | `0.079038048` | `0.039761781` |
| `best.pt` SHA-256 | `6e51a092de9b0553aa7d85bd1432a394425ef39059d7bbcf21bc7e57844273f1` | `caf71b29d848cf9bc02d305999d645b59af1f16c0a9e6bfebd0b540f2f08861b` |

Checkpoint selection rule, identical for both: **lowest macro-averaged normalized MSE on the
full validation split**.

### The two runs used different code commits — verified immaterial

`docs/EXPERIMENT_PLAN.md` requires both models to run with the same data manifest and primary
seed, which they did. They were launched from different commits because work continued between
launches. That was checked rather than assumed:

```
git diff --name-only 077d6b53 da865691 -- swe_sr/train.py swe_sr/training/ \
    swe_sr/models/ swe_sr/data/ swe_sr/metrics/ configs/
  -> swe_sr/data/fresh.py
```

`swe_sr/data/fresh.py` was **added** in that range and is imported only by
`swe_sr.evaluate_fresh`, never by training. `train.py`, both model definitions, the data
pipeline, the metrics used for the loss, and every config are byte-identical across the two
commits. The remaining differences are evaluation, reporting, tests, and CI. The comparison is
therefore clean.

Note also that `summary.json` for these two runs records `git_commit` as HEAD at *write* time
rather than launch time, because commits landed while the runs were in flight. The launch
commits above, taken from each `run_id`, are the accurate ones. `swe_sr/train.py` now records
`git_commit` at launch and `git_commit_at_completion` separately so this cannot recur.

## Frozen metric definitions

From `docs/VALIDATION.md`, and unchanged by this freeze:

- **Training objective and selection metric**: equal-weight normalized channel MSE, macro
  averaged over `[eta, u, v]` (D005).
- **Aggregation**: per snapshot, then within trajectory, then equal weight across trajectories;
  95% percentile bootstrap resampling *trajectories*, not snapshots.
- **relL2 and mass error**: per-snapshot ratios reduced by the same protocol. Never averaged
  per batch — that made the reported value depend on batch size, and was corrected under V-05.
- **Physical diagnostics**: computed after de-normalization, in SI, per D014.
- **Reference**: normalized channels have unit variance, so predicting the channel mean scores
  exactly 1.0.

## Frozen results — held-out test split (1,576 snapshots, 8 trajectories)

This table is a **verified** block: `python -m swe_sr.docgen check` re-renders it from
`docs/results/index.json` and fails if it no longer matches, but never rewrites it. Rewriting a
freeze to agree with whatever the artifacts currently say would destroy the only property it
exists to provide. A mismatch means a new decision and a new freeze, not an edit here.

<!-- BEGIN verified: results:frozen-test -->
| Method | Params | normMSE | 95% CI | eta relL2 | u relL2 | v relL2 | mass err |
|---|---:|---:|---|---:|---:|---:|---:|
| nearest | 0 | 0.4301 | [0.3076, 0.5435] | 0.6305 | 0.6987 | 0.7044 | 0.0439 |
| bicubic | 0 | 0.4295 | [0.3069, 0.5431] | 0.6239 | 0.6983 | 0.7021 | 0.0392 |
| EDSR | 1,517,571 | 0.0830 | [0.0543, 0.1129] | 0.2826 | 0.3168 | 0.3150 | 0.0540 |
| U-Net | 1,930,208 | **0.0400** | [0.0261, 0.0544] | **0.1951** | **0.2149** | **0.2135** | **0.0347** |
<!-- END verified: results:frozen-test -->

Paired bootstrap against bicubic, both excluding zero: EDSR `-0.3466 [-0.4334, -0.2514]`,
U-Net `-0.3895 [-0.4920, -0.2805]`.

Both independently recomputed from the stored arrays by `scripts/verify_independent.py`, which
reimplements normalization, destaggering, every field metric, the mass diagnostic, and the
aggregation protocol in plain numpy from the specification: **20 checks each, all passing**.

## Promotion to the canonical path, 2026-08-13

The release was moved from `data/staging/` to `data/{raw,processed}/` on owner instruction. Only
the **location** changed; nothing frozen did. Re-verified after the move:

- processed manifest hash is `af02e44f...`, identical to the frozen value above;
- `python -m swe_sr.data.validate` passes 12/12 gates at the canonical path;
- both frozen runs re-evaluate to identical numbers, U-Net 0.0400 and bicubic 0.4295.

Because location is informational rather than frozen content, this is not a superseding change
and no new freeze is required. Two details of how it was done matter for provenance:

- The superseded development dataset was **moved aside, not deleted**, to
  `data/raw/swe_gaussian_32x128_v1.superseded-dev-run-5a60ecc-dirty`. It was never the frozen
  release: commit `5a60ecc...-dirty`, manifest hash `3a65583b...`, and no normalization block.
  Keeping it rather than destroying it leaves the distinction auditable; it can be removed at any
  time.
- `data/staging/{raw,processed}/swe_gaussian_32x128_v1` are now **symlinks** to the canonical
  paths. Both frozen runs record `data/staging/processed/.../manifest.json` in their own
  `config.yaml`, and editing a completed run's config to match a later move would falsify its
  provenance. The symlinks keep those recorded paths resolvable without touching the artifacts.

## Outstanding against this freeze

- The backup 64->256 pair (D-06, O-01) remains deferred and is outside this freeze.
