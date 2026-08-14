# Reproducing the whole experiment from scratch

Split out of `README.md` (D024). `docs/PIPELINE.md` explains what each stage does and why;
this is the sequence to run, with measured timings and the things a rerun will not
reproduce.

Nothing in this repository needs to be preserved to reproduce the science: the whole primary
experiment is about **3.5 hours** end to end on one 16-thread CPU allocation — a few minutes of
data generation, ~15 minutes of staged plumbing and cost checks, 2.7 hours of training for both
models, and well under an hour of scoring and figures. The optional cross-resolution transfer adds
a 5.3 GB dataset and another round of evaluation.

## What is disposable

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

## The full sequence

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

## What a rerun does and does not reproduce

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
