# shallow-water-super-resolution

Neural spatial super-resolution for a 2D shallow-water solver. A coarse numerical solution is
the model input; an **independently integrated** fine-grid solution at the same physical time is
the target:

```
F_theta: R^(3 x 32 x 32)  ->  R^(3 x 128 x 128)
```

Channels are `[eta, u, v]` — surface elevation and depth-averaged velocity. Total water height
is `h = H + eta`; `eta` is never renamed to `h`.

The coarse input is a genuine coarse PDE solve, not a downsampled fine field (D002). That is the
point of the study, and it has a consequence worth knowing before reading any number: the two
solves progressively decorrelate, so baseline error grows strongly with lead time and every
result is reported stratified by it.

## Results

The frozen primary experiment is `docs/EXPERIMENT_FREEZE.md` (T-03, D020); generated results are
in `docs/RESULTS.md`. Held-out test split, 1,576 snapshots over 8 trajectories, normalized
macro-averaged MSE with a 95% trajectory bootstrap interval:

| Method | Params | normMSE | 95% CI |
|---|---:|---:|---|
| nearest | 0 | 0.4301 | [0.3076, 0.5435] |
| bicubic | 0 | 0.4295 | [0.3069, 0.5431] |
| EDSR x4 | 1,517,571 | 0.0830 | [0.0543, 0.1129] |
| U-Net x4 | 1,930,208 | **0.0400** | [0.0261, 0.0544] |

Normalized channels have unit variance, so predicting the channel mean scores exactly 1.0.

Three findings beyond the headline:

- **U-Net beats EDSR by 2.1x** in distribution, for 1.27x the parameters and about 1.9x the
  compute — 79.8 against 42.0 samples/s, averaged over the epochs of a 30,000-step run on a fixed
  16-thread allocation. A 2,000-step pilot had them tied, so this only emerged at 30,000 steps.
- **Lower pixel error does not imply better physics.** U-Net improves relative mass error over
  bicubic (0.0347 against 0.0392) while EDSR makes it *worse* (0.0540) despite beating bicubic
  fivefold on MSE.
- **The ranking inverts out of distribution.** On the unseen annular `ring_ood` family, EDSR
  scores 0.3246 against U-Net's 0.3505, so U-Net's in-distribution advantage does not transfer.
  It inverts again under a *resolution* shift — see below.

### Cross-resolution transfer (`docs/TRANSFER.md`)

The same two checkpoints, evaluated with no retraining on a newly generated 64→256 pair. Skill
against bicubic on the same data falls from 5.2× to 2.4× (EDSR) and 10.7× to 2.2× (U-Net), and
EDSR wins — paired `+0.00533 [+0.00268, +0.00826]`, U-Net worse on 8 of 8 trajectories.

The sharper finding is stratified: **both models lose to bicubic below about 12.7 h of lead time**,
by up to 73×, and they do so on the trained pair too, which the aggregate had concealed. An exact
decomposition `MSE_model/MSE_bicubic = r² − 2rc + 1` shows the cause is correction *magnitude*
(r up to 9.5 where 1 is right) rather than direction, so it is recoverable in principle while the
alignment loss that accounts for most of the skill drop is not.

## Getting started

```bash
git clone --recurse-submodules <this repo>        # the solver reference is a pinned submodule
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,viz]"
./scripts/check.sh                                # lint, format, types, fast tests
```

`scripts/check.sh` is the single gate; CI runs exactly it, so local and CI cannot drift.

Nothing is downloaded at run time: every dataset, checkpoint, and figure below is produced on
this machine from seeds and configs. `data/`, `runs/`, and `viz/` are gitignored, so a fresh
clone starts with no data at all.

## How the pipeline works

Six stages, each with one entry point and one immutable output. The rule that shapes all of them
is D002: the coarse input is its own PDE solve, never a downsampling of the fine target.

```
 data/registries/ic_registry_v1.json        48 analytic ICs, seeds, stable IDs, fixed splits
 configs/data/<pair>.yaml                   grids, spin-up, save cadence
              │
              ▼  swe_sr.data.generate       two independent solves per IC
 data/raw/<id>/trajectories/*.h5            physical SI units, C-grid staggered
 data/raw/<id>/manifest.json                configs, times, coord hashes, checksums
 data/processed/<id>/manifest.json          the same + train-split normalization
              │
              ▼  swe_sr.data.validate       12 gates, all recomputed from the arrays
              │
 configs/model/<arch>.yaml                  architecture
 configs/experiment/<stage>.yaml            schedule
              │
              ▼  swe_sr.train               loader destaggers + normalizes here
 runs/<run_id>/{config.yaml, environment.json, dataset_manifest.json,
                metrics.csv, summary.json, curves.png, checkpoints/{best,last}.pt}
              │
              ▼  swe_sr.evaluate            model + nearest + bicubic on identical states
 runs/<run_id>/evaluation_test.json
 runs/<run_id>/evaluation_fresh_<scenario>.json        swe_sr.evaluate_fresh
              │
              ▼  swe_sr.report              composes, never recomputes
 docs/RESULTS.md          scripts/verify_independent.py — recomputed in plain numpy
```

### 1. Initial conditions — one immutable registry

`swe_sr/data/registry.py` draws 48 analytic initial conditions deterministically from the seed
IDs 0–47 and writes `data/registries/ic_registry_v1.json`. Each is a sum of one to three Gaussian
surface bumps (signed amplitude 0.5–1.5 m, width 65–120 km, centred with a 2σ wall margin, `u = v = 0`),
redrawn if it would violate positive depth or the wall margin.

Two properties are enforced in code rather than by convention:

- **The split is fixed before any simulation runs** (D004). Seeds 0–31 are train, 32–39
  validation, 40–47 test — a pure function of the seed, so no shuffle can leak a trajectory.
- **Trajectory IDs are content-derived** (UUID5 over registry version and seed), so the same
  seed yields the same ID on any machine and reordering the file cannot renumber anything.

This is the only shared artifact between the two resolution pairs (D008) and the only thing under
`data/` that is tracked in Git. `tests/data/test_registry.py` pins its SHA-256, so a change to the
sampling code is a visible test failure rather than a silent renumbering.

### 2. Paired data generation — two independent solves per trajectory

`python -m swe_sr.data.generate --config configs/data/primary_32x128.yaml`

For every registry entry, `swe_sr/data/generate.py` evaluates the *same analytic* IC on the coarse
grid and on the fine grid and integrates each with `swe_sr.solver` (a transcription of the pinned
`references/shallow-water/swe.py`: closed 1,000 km basin, rotating beta-plane, H = 100 m, no
friction or wind). Both members use the **fine** grid's CFL time step (D003), so their saved-time
arrays are bit-identical.

| | primary `swe_gaussian_32x128_v1` | backup `swe_gaussian_64x256_v1` |
|---|---|---|
| grids | 32² → 128² | 64² → 256² |
| shared `dt` | 25.139798 s | 12.520605 s |
| spin-up discarded | 288 steps | 576 steps |
| save stride / count | every 24, ×197 | every 48, ×197 |
| last step / duration | 4992 / 34.86 h | 9984 / 34.72 h |
| on disk (gzip-4 HDF5) | 1.5 GB | 5.3 GB |

The stride and step cap double for the backup precisely because its `dt` halves, which keeps the
two pairs at matching physical times and frame counts (D017). Note the grids are ×4 in *node
count*, not exact ×4 meshes: both endpoints are included, so the spacing ratios are 127/31 and
255/63.

Before anything is written, the generator asserts each trajectory is admissible (finite, positive
total depth, mass drift at roundoff) and that the two members' saved times are equal. It then
writes, per trajectory, one HDF5 file with frame-sized chunks holding `lr`, `hr`, times,
coordinates, and metadata — in **physical SI units, staggered on the Arakawa C-grid as the solver
produced them**. Normalization never touches stored data.

Two manifests follow. `data/raw/<id>/manifest.json` records the resolved solver configs, shared
time step, saved steps and times, coordinate hashes, IC parameters, split membership, and a
per-array checksum. `data/processed/<id>/manifest.json` is the same plus the **normalization
block**: one mean and standard deviation per channel, fitted on the *train split only*, on the
*fine* fields, *after destaggering* (D019), so the statistics match what models actually consume.
Both refuse to overwrite a manifest with a different hash — releases are immutable.

`python -m swe_sr.data.validate --manifest data/processed/<id>/manifest.json` is the gate. It
recomputes 12 checks **from the arrays**, not from what the generator logged: every array checksum
(384 for a 48-trajectory release — 8 arrays each), shapes, coincident endpoints, coordinate hashes,
exact within-pair saved times, finiteness and positive depth, exactly-zero wall velocities, split
disjointness, agreement with the registry, mass drift under 1e-12, and the train-split
normalization re-fitted and matched to 1e-9. It takes about 40 s on the primary release and exits
non-zero if any gate fails, so it works as a pre-training gate.

### 3. What one training sample is

`swe_sr/data/dataset.py` (`PairedSnapshotDataset`) is the only place normalization happens. One
sample is **one snapshot pair**, drawn independently:

1. read frame `t` of `lr` and of `hr` from HDF5 — a single chunked read, so an epoch never
   materializes a trajectory;
2. **destagger** (D011): average `u` along x and `v` along y onto cell centres, halving the
   wall-adjacent column/row, so all three channels share the `eta` grid and endpoint-aligned
   interpolation is exact for every channel;
3. **normalize** with the manifest's train-split statistics — `check_pair_id` refuses to apply one
   pair's statistics to another;
4. cast to float32 and return `{coarse [3,32,32], fine [3,128,128], trajectory_id, frame, time}`.

Full frames, never patches. Time is not a model input — the contract is a single instant,
`[B,3,H,W] → [B,3,4H,4W]`; `time` rides along only for pairing, lead-time stratification, and
provenance. Sample counts: 6,304 train / 1,576 validation / 1,576 test pairs.

**Augmentation is implemented and off by default** (D018). Reflections with the correct velocity
sign flip, and the transpose that swaps `u` with `v`, are geometrically exact on cell-centred
fields — but they are not symmetries of a rotating beta-plane. `augmentation_symmetry_error()`
measures the discrepancy: about 0.9 relative with rotation on, and reflections stay broken at
0.048 even with rotation off because C-grid staggering places `u` on east faces. They remain
available via config as a future ablation.

### 4. The two models

Both share the outer form fixed by D006, so the comparison isolates the learned residual:

```
y_hat = bicubic(x) + R_theta(x)        # bicubic: mode="bicubic", align_corners=True
```

One `upsample()` in `swe_sr/models/common.py` serves the models' residual path *and* the bicubic
baseline, so a comparison between them cannot accidentally measure an interpolation difference.
`align_corners=True` is required, not stylistic: the grids are `linspace(-L/2, L/2, N)` with both
endpoints, so corner pixel centres sit on the domain corners at every resolution. A model with
all weights zeroed reproduces the bicubic baseline exactly, which is what makes any reported gain
attributable to the learned residual — `tests/models/test_models.py` asserts it.

**Residual U-Net ×4** — 1,930,208 parameters, `configs/model/unet_x4.yaml`. Bicubic-upsample to
128² *first*, then work at full resolution: bias-free 3×3 head to 32 features; three encoder
stages of 32/64/128 features, two residual blocks each with SiLU; 2×2 average pooling between
stages (a fixed low-pass filter, per the 2024 dynamic shallow-water paper); a pixel-shuffle ×2
decoder concatenating the matching encoder features; bias-free 3×3 tail to three residual
channels.

**EDSR ×4** — 1,517,571 parameters, `configs/model/edsr_x4.yaml`. Does its work on the *coarse*
grid and upsamples at the end: 3×3 head to 64 features; 16 residual blocks with ReLU and
`res_scale = 0.1`; a body tail convolution closed by a global skip; two pixel-shuffle ×2 stages;
3×3 tail. Three deliberate departures from the reference EDSR-PyTorch: no RGB `MeanShift` (dataset
normalization replaces it), random initialization (natural-image weights do not transfer to
rotating fields), and the outer bicubic residual for parity with the U-Net.

That structural contrast — refine at full resolution versus extract features coarse and upsample —
is what the comparison tests. The parameter counts are within 1.27×, so it is about architecture
rather than capacity. Both are resolution-generic: one set of weights accepts 32²→128² and
64²→256², which is what made the transfer experiment possible.

### 5. Training

```bash
python -m swe_sr.train --config configs/model/unet_x4.yaml --experiment configs/experiment/full.yaml
```

`--config` picks the architecture; `--experiment` picks the schedule (`smoke`, `pilot`, `full`,
`full_diagnostic`). Without `--experiment` the built-in defaults in `swe_sr/training/config.py`
are the full schedule.

| Setting | Value |
|---|---|
| loss | equal-weight normalized channel MSE over `[eta, u, v]` (D005), reduced in fp32 even under autocast |
| optimizer | AdamW, lr 1e-4, weight decay 1e-6 |
| schedule | 500-step linear warmup, then cosine decay to zero over `max_steps` |
| batch / clip | 8 samples, global grad-norm clip 1.0 |
| budget | 100 epochs or 30,000 steps, whichever first; 788 steps per epoch |
| early stopping | 15 epochs without validation improvement |
| precision | BF16 autocast if `avx512_bf16` is present, else FP32 — the resolved choice is recorded |
| seed | 20260812 |
| checkpoint rule | lowest macro-averaged normalized MSE on the **full** validation split |

Three properties are treated as requirements:

- **Determinism.** Batch order is `random.Random(f"{seed}:{epoch}")`, a pure function of
  `(seed, epoch)`, so a rerun with the same config reproduces the same curve. There is no
  `DataLoader` — a plain generator is used precisely to keep that guarantee.
- **Selection on the whole validation split**, accumulated as a sample-weighted sum so a short
  final batch cannot skew it. Never a sampled subset.
- **Honest provenance.** The commit is captured at *launch* (`git_commit`) and again at write time
  (`git_commit_at_completion`), with a `-dirty` marker if the tree was not clean.

Every epoch writes `metrics.csv`, redraws `curves.png`, saves `checkpoints/last.pt`, and saves
`checkpoints/best.pt` when validation improves — so a killed run is still evaluable. Validation
error is also recorded **stratified by lead time**, against a bicubic reference computed once, in
`summary.json`. That is the artifact behind every stratified claim in the docs.

Training runs on the Slurm CPU partition (D015 — no CUDA device is reachable anywhere):

```bash
cbrun -t rocky -- srun -c 16 -- python -m swe_sr.train \
    --config configs/model/unet_x4.yaml --experiment configs/experiment/full.yaml
```

Measured for the two frozen runs, 30,000 steps, BF16 on AMD EPYC 9R14 with 16 threads (Slurm jobs
296035 and 296390, node `cpu-dy-x48-m7a-7`): EDSR **0.92 h** wall clock and 700 MB peak RSS;
U-Net **1.73 h** and 989 MB. Throughput over the same allocation was 79.8 against 42.0 samples/s
(`summary.json` → `projection.mean_samples_per_second`), the ~1.9× compute ratio quoted above.
`torch.set_num_threads` follows `torch_threads` in the config; leave it at 0 to inherit torch's
default and let `-c` govern.

### 6. Scoring

```bash
python -m swe_sr.evaluate --run-dir runs/<run-id>              # held-out test split
```

`swe_sr/evaluate.py` loads the run's own `best.pt` and its own frozen manifest — the dataset is
read from the run's `config.yaml`, so a run cannot be silently scored against different data — and
evaluates the model **and both baselines on identical states** through the same code path. It
refuses to fit anything: normalization comes from the frozen manifest, so the test split never
influences a statistic.

The aggregation protocol (`docs/VALIDATION.md`) is: metric per snapshot → mean within trajectory →
**equal weight across trajectories**, with a 95% percentile bootstrap that resamples
*trajectories*, not snapshots. Ratio metrics (relL2, relative mass error) follow the same route
rather than being averaged per batch — a batch mean makes a norm ratio depend on batch size, a
defect the independent verifier actually caught. Physical diagnostics are computed after
de-normalization, in SI (D014). Reported per method: parameters, normalized macro MSE with CI,
per-channel MSE/RMSE/relL2/max-abs, relative mass error, ms/frame, the per-trajectory means (so a
paired test across two separate runs is possible at all), and the full lead-time breakdown. Every
model is also paired-bootstrapped against bicubic on the same trajectories.

The lead-time breakdown is not optional colour. Because the coarse and fine solves are independent
integrations, they decorrelate: bicubic goes 0.0090 at 2 h to 0.9759 at 34.1 h on the test split.
A single aggregate therefore summarizes a mixture of a well-posed regime and a partly
unpredictable one, and reading only the aggregate hides that both models *lose* to bicubic at
short lead times.

```bash
python -m swe_sr.evaluate_fresh --run-dir runs/<run-id> --scenario fresh_id   # new Gaussian seeds
python -m swe_sr.evaluate_fresh --run-dir runs/<run-id> --scenario ring_ood   # unseen annular family
```

Fresh workloads are generated on demand from the scenario name, from a reserved seed band at
10,000+ that `assert_disjoint_from_registry` proves cannot overlap training, and are solved by the
same two-independent-solves route as the training data. Normalization comes from the *training*
manifest, never refitted. They default to the training manifest's own snapshot count so the
lead-time span is comparable, and they are reported separately — never folded into the test score.

```bash
python -m swe_sr.report --runs runs/<edsr-run> runs/<unet-run> --out docs/RESULTS.md
python scripts/verify_independent.py --run-dir runs/<run-id> --split test
```

`swe_sr.report` composes `docs/RESULTS.md` from the evaluation JSONs only — it recomputes nothing,
so every number is traceable to the artifact that produced it, and it reads
`docs/EXPERIMENT_FREEZE.md` to identify frozen runs rather than trusting a run's own stage label.
`scripts/verify_independent.py` is the real check: it reimplements normalization, destaggering,
every field metric, the mass diagnostic, and the aggregation protocol in plain numpy from the
written specification, importing only the HDF5 reader and the model definition. Both frozen runs
pass 20 checks each.

## Reproducing the whole experiment from scratch

Nothing in this repository needs to be preserved to reproduce the science: the whole primary
experiment is about **3.5 hours** end to end on one 16-thread CPU allocation — a few minutes of
data generation, ~15 minutes of staged plumbing and cost checks, 2.7 hours of training for both
models, and well under an hour of scoring and figures. The optional cross-resolution transfer adds
a 5.3 GB dataset and another round of evaluation.

### What is disposable

Everything below is regenerable from tracked configs and seeds. Sizes are as measured on this
checkout.

| Path | Size | Regenerate with |
|---|---:|---|
| `data/raw/swe_gaussian_32x128_v1/`, `data/processed/…` | 1.5 GB | `generate --config configs/data/primary_32x128.yaml` |
| `data/raw/swe_gaussian_64x256_v1/`, `data/processed/…` | 5.3 GB | `generate --config configs/data/backup_64x256.yaml` |
| `data/raw/swe_gaussian_32x128_smoke/`, `…_64x256_smoke/` | 36 MB | the two `*_smoke.yaml` data configs |
| `data/raw/swe_gaussian_32x128_v1.superseded-dev-run-5a60ecc-dirty/` | 1.5 GB | **delete outright** — never a release (dirty commit, no normalization block); kept only so the distinction stayed auditable |
| `runs/` — 11 smoke/pilot runs | ~130 MB | the `smoke` and `pilot` experiment configs (mostly checkpoints) |
| `runs/` — the 2 frozen runs | 29 MB | the `full` schedule, ~2.7 h combined on 16 threads |
| `viz/` | 78 MB | the four `scripts/plot_*.py` / `visualize.py` entry points |
| `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `*.egg-info/` | small | recreated by `./scripts/check.sh` |

Two things under `data/` are **not** ordinary generated output:

- `data/registries/ic_registry_v1.json` is tracked in Git and is the contract every manifest
  references by hash. Deleting it is recoverable (`git checkout`, or the generator rebuilds an
  identical one from seeds), but never hand-edit it — `load_registry` validates it on every read.
- `data/staging/{raw,processed}/swe_gaussian_32x128_v1` are **symlinks** into the canonical paths.
  The two frozen runs record `data/staging/processed/…/manifest.json` in their own `config.yaml`,
  and a completed run's provenance is not rewritten to match a later file move. Delete these and
  re-evaluating *those two existing runs* breaks; restore with
  `ln -s ../../processed/swe_gaussian_32x128_v1 data/staging/processed/swe_gaussian_32x128_v1`
  (and the same under `raw/`). Runs you train yourself record the canonical path and need none of
  this.

**Delete before regenerating.** Manifests are immutable: `manifest.write` refuses to replace one
whose hash differs, and it is called *after* all 48 trajectories have been solved. Regenerating on
top of an existing release therefore fails at the very end, wasting the whole run. Remove the
dataset directory first:

```bash
rm -rf data/raw/swe_gaussian_32x128_v1 data/processed/swe_gaussian_32x128_v1
```

### The full sequence

Timings are on the Slurm CPU partition, 16 threads, BF16 on AMD EPYC 9R14.

```bash
# 0. Environment and gate ------------------------------------------------- ~2 min
git submodule update --init                        # the solver reference; parity tests need it
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,viz]"
./scripts/check.sh                                 # must exit 0 before anything else

# 1. Data: smoke first, then the primary release ------------------------- ~4 min
python -m swe_sr.data.generate --config configs/data/primary_32x128_smoke.yaml   # 6 traj, ~3 s
python -m swe_sr.data.validate --manifest data/processed/swe_gaussian_32x128_smoke/manifest.json
python -m swe_sr.data.generate --config configs/data/primary_32x128.yaml         # 48 traj, ~2.5 min
python -m swe_sr.data.validate --manifest data/processed/swe_gaussian_32x128_v1/manifest.json
# ~40 s; expect 12/12 gates, 384 checksums recomputed (48 trajectories x 8 arrays),
# worst relative mass drift 1.067e-14, minimum total depth 98.6689 m, and the normalization
# in docs/EXPERIMENT_FREEZE.md re-derived: eta mean +1.758229e-02, std 1.761697e-01

# 2. Plumbing and cost check --------------------------------------------- ~15 min
for model in edsr unet; do
  # smoke: 4 steps, seconds, locally -- shapes, gradients, checkpoints, run directory
  python -m swe_sr.train --config configs/model/${model}_x4.yaml \
      --experiment configs/experiment/smoke.yaml
  # pilot: 1,970 steps on 8 train trajectories -- measures cost, ~4 min EDSR / ~7 min U-Net
  cbrun -t rocky -- srun -c 16 -- python -m swe_sr.train \
      --config configs/model/${model}_x4.yaml --experiment configs/experiment/pilot.yaml
done
# each pilot prints projected_full_run_hours; confirm it fits the budget before step 3

# 3. The two full runs ------------------------------- EDSR ~0.92 h, U-Net ~1.73 h
for model in edsr unet; do
  cbrun -t rocky -- srun -c 16 -- python -m swe_sr.train \
      --config configs/model/${model}_x4.yaml --experiment configs/experiment/full.yaml
done
# note the two run IDs it prints; everything below takes them as --run-dir

# 4. Score --------------------------------------------------------- minutes each
for run in runs/<edsr-run> runs/<unet-run>; do
  python -m swe_sr.evaluate       --run-dir $run --split test
  python -m swe_sr.evaluate_fresh --run-dir $run --scenario fresh_id
  python -m swe_sr.evaluate_fresh --run-dir $run --scenario ring_ood
  python scripts/verify_independent.py --run-dir $run --split test   # must report all checks passing
done
python -m swe_sr.report --runs runs/<edsr-run> runs/<unet-run> --out docs/RESULTS.md
python scripts/plot_final.py --runs runs/<edsr-run> runs/<unet-run>

# 5. Optional: cross-resolution transfer ------------- generation is ~8x step 1's solver work
python -m swe_sr.data.generate --config configs/data/backup_64x256.yaml
python -m swe_sr.data.validate --manifest data/processed/swe_gaussian_64x256_v1/manifest.json
for run in runs/<edsr-run> runs/<unet-run>; do
  python -m swe_sr.evaluate --run-dir $run --split test \
      --manifest data/processed/swe_gaussian_64x256_v1/manifest.json
  python scripts/analyze_transfer.py --run-dir $run --stride 4 \
      --manifest data/processed/swe_gaussian_64x256_v1/manifest.json \
      --out $run/decomposition_test__swe_gaussian_64x256_v1.json
done
python scripts/plot_transfer.py --runs runs/<edsr-run> runs/<unet-run>
```

A cross-pair evaluation writes `evaluation_test__<pair_id>.json`, never the canonical
`evaluation_test.json`. The name is derived from the data rather than passed as a flag, because
the earlier behaviour would have overwritten a frozen result with transfer numbers.

One deliberate difference from the frozen runs: step 3 above uses `configs/experiment/full.yaml`,
while the two frozen runs were launched with `full_diagnostic.yaml`. **Every schedule value in the
two files is identical** — only the `stage` label differs, and the frozen runs keep theirs because
a completed run's provenance is not rewritten to match a later decision (see the honesty note in
`docs/EXPERIMENT_FREEZE.md`). A rerun should use `full.yaml`.

### What a rerun does and does not reproduce

- **Reproduced exactly:** the IC registry hash
  `976e3a577a25a633c6a2625263f23e60482768965029805a5efd16be97ab7c8c`. This was verified from a
  fresh clone in a fresh venv (G8 evidence in `TASKS.md`), and it is the check that the dataset is
  reproducible *from source* rather than merely regenerable.
- **Not reproduced:** run IDs and checkpoint digests. A run ID is
  `<timestamp>_<model>_<config8>_<commit8>`, and the config hash covers the manifest path — so
  regenerating at the canonical path rather than the frozen runs' `data/staging/…` path yields a
  different hash by design. The frozen `best.pt` SHA-256 digests in `docs/EXPERIMENT_FREEZE.md`
  pin *those* artifacts; they are not a target for a rerun.
- **Not reproduced:** the manifest hash `af02e44f…`. `manifest_hash` covers
  `provenance.git_commit`, so a dataset generated at any other commit hashes differently even with
  byte-identical arrays. Use the registry hash and the 12 validation gates as the equivalence
  check, not the manifest hash.
- **Expect small numerical drift** on different hardware or a different torch build. BF16 autocast
  engages only where `avx512_bf16` exists and falls back to FP32 otherwise — the run records which
  it resolved to, and `ms/frame` is host-load dependent and not a controlled cross-run comparison.
- **The frozen results are not overwritten by any of this.** `docs/EXPERIMENT_FREEZE.md` is the
  record; changing what it pins requires a new decision record and a new freeze, not an edit.

## Command reference

```bash
python -m swe_sr.data.generate  --config configs/data/primary_32x128.yaml
python -m swe_sr.data.validate  --manifest data/processed/swe_gaussian_32x128_v1/manifest.json
python -m swe_sr.train          --config configs/model/unet_x4.yaml \
                                --experiment configs/experiment/full.yaml
python -m swe_sr.evaluate       --run-dir runs/<run-id>
python -m swe_sr.evaluate_fresh --run-dir runs/<run-id> --scenario ring_ood
python -m swe_sr.report         --runs runs/<edsr-run> runs/<unet-run> --out docs/RESULTS.md

python scripts/verify_independent.py --run-dir runs/<run-id>  # recomputes metrics from arrays
python scripts/visualize.py --seed 0 --all                    # animations via reference viz_tools
python scripts/plot_pilot.py --runs runs/<run> ...            # curves and the lead-time figure
python scripts/plot_final.py --runs runs/<edsr-run> runs/<unet-run>   # the final result figures
python scripts/analyze_transfer.py --run-dir runs/<run> --manifest <other-pair>  # r/c decomposition
python scripts/plot_transfer.py --runs runs/<edsr-run> runs/<unet-run>           # transfer figures
```

The figures land in `viz/`, which is gitignored — they are regenerated from the frozen
checkpoints rather than committed. `docs/RESULTS.md` lists what each one shows.

Any of these may be prefixed with `cbrun -t rocky -- srun -c 16 --` to run on the Slurm CPU
partition (D015 — no CUDA device is reachable); training is the only one where it matters.

## Documentation

Read in this order. `CLAUDE.md` lists the same set and states the project's non-negotiable rules.

| File | Contents |
|---|---|
| `docs/PROJECT_SPEC.md` | goal, scope, research questions, definition of done, open questions |
| `docs/DATASET.md` | resolution pairs, IC family, splits, normalization, storage contract |
| `docs/ARCHITECTURE.md` | repository layout, batch contract, both model designs, run directory |
| `docs/EXPERIMENT_PLAN.md` | baselines, loss, optimizer schedule, staged runs, artifacts |
| `docs/VALIDATION.md` | data gates, metric definitions, aggregation protocol, negative tests |
| `docs/DECISIONS.md` | every project decision, D001 to D021, with reasons and consequences |
| `docs/EXPERIMENT_FREEZE.md` | the frozen T-03 record: hashes, commits, seeds, checkpoint digests |
| `docs/RESULTS.md` | generated comparison, lead-time breakdown, fresh workloads, limitations |
| `docs/TRANSFER.md` | the 64->256 cross-resolution transfer test and the r/c decomposition |
| `docs/ABLATION_RESIDUAL.md` | ablation 3 (D022): outer bicubic residual against direct prediction |
| `docs/SIGNOFF.md` | the I-03 audit: what is verified, what is outstanding |
| `TASKS.md` | task status and per-gate evidence |

## Verification posture

Numbers here are recomputed independently before being reported.
`scripts/verify_independent.py` reimplements normalization, destaggering, every field metric, the
mass diagnostic, and the aggregation protocol in plain numpy from the specification, importing
only the HDF5 reader and the model definition. It found a real defect that way — an evaluation
metric whose value depended on batch size — which reusing the package's own metrics could not
have caught.

`docs/SIGNOFF.md` records what is verified and what is not, including known limitations.
