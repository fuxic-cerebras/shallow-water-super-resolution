# How the pipeline works

Split out of `README.md`, which is a front door rather than a manual (D024). This is the
walkthrough: what each stage consumes, what it writes, and which rule fixes the choice.
`docs/REPRODUCE.md` is the command-by-command rerun; the contracts these stages implement
are `docs/DATASET.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PLAN.md`, and
`docs/VALIDATION.md`.

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

## 1. Initial conditions — one immutable registry

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

## 2. Paired data generation — two independent solves per trajectory

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

## 3. What one training sample is

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

## 4. The three models

All three share the outer form fixed by D006, so the comparison isolates the learned residual:

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

**ConvMixer ×4** — 1,720,067 parameters, `configs/model/convmixer_x4.yaml` (D023). *Isotropic*:
one resolution end to end, no pooling anywhere. A `patch_size=1` stem lifts to 256 features, then
16 identical blocks, each a 9×9 **depthwise** convolution with a residual followed by a pointwise
1×1, with GELU and BatchNorm after each; then a 1×1 projection to 64 channels and EDSR's decoder
(two pixel-shuffle ×2 stages, 3×3 tail). `k=9` and `p=1` come from the paper's CIFAR-10 ablations,
which are run at 32×32 — exactly this project's low resolution.

That gives it a **129 px receptive field at low resolution, with no downsampling** — wider than
both the 32² and 64² grids, so one unit integrates the whole basin. EDSR's 16 stacked 3×3 blocks
reach only ~33 px, barely one grid width, and the U-Net reaches basin scale only by pooling. The
three models therefore separate three distinct inductive biases — pyramid, stacked-small-kernel,
and isotropic-large-kernel — rather than three variations on one. That is what the comparison
tests, at 1.52 M / 1.72 M / 1.93 M parameters, so it is about architecture rather than capacity.

Two caveats stated up front. ConvMixer is the one model that **keeps BatchNorm**, contradicting
EDSR's published finding that normalization harms super-resolution; D023 records why the arm
keeps it rather than resolving the disagreement by assumption. Because BatchNorm couples samples
while training, every path that reports a number calls `model.eval()` first, which restores exact
per-sample independence. And large-kernel depthwise convolution is slow on CPU — this arm costs
roughly 3× EDSR's wall clock per step.

All three are resolution-generic: one set of weights accepts 32²→128² and 64²→256², which is what
made the transfer experiment possible.

## 5. Training

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

## 6. Scoring

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
