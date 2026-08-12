# Dataset Design

## Resolution family

Define two independently reproducible paired datasets:

| Pair ID | Role | LR grid | HR grid | Shape factor | Raw payload |
|---|---|---:|---:|---:|---:|
| `swe_gaussian_32x128_v1` | Primary | 32 x 32 | 128 x 128 | x4 | 1.840 GiB |
| `swe_gaussian_64x256_v1` | Backup | 64 x 64 | 256 x 256 | x4 | 7.359 GiB |

For each pair, evaluate the same analytic initial condition directly on LR and HR
coordinates and integrate both states independently with the existing `swe.py` scheme.
No LR array is obtained by resizing an HR array. Each pair uses one time step stable
for its HR grid, giving its LR and HR members bit-identical saved-time arrays. The two
pair variants may use different time steps and must not be mixed in one training run.

Each pair runs to the same physical duration, about 34.9 h, which matches the 4999
updates the reference `swe.py` performs on the primary pair's time step (D017).

With nominal resting depth, the shared steps are approximately 25.1 s for the primary
pair and 12.5 s for the backup pair. The generator calculates and validates the actual
CFL bound from the resolved state rather than hard-coding either estimate. For the
primary pair,

$$
\Delta t = 0.1\frac{\min(\Delta x_{HR},\Delta y_{HR})}{\sqrt{gH}}
\approx 25.1\ \mathrm{s}.
$$

These are x4 node-count shapes, not exactly nested 4x meshes, because `swe.py`
includes both endpoints. The spacing ratios are `127/31 = 4.0968` and
`255/63 = 4.0476`. Store coordinates and their hashes and use endpoint-aligned
interpolation; do not describe the grids as exact fourfold mesh refinements.

## Height convention

`swe.py` evolves surface elevation `eta`, while the physical water-column height is
`h = H + eta`. Store `eta`, `u`, and `v`; store `H` in metadata. This avoids spending
model dynamic range on a constant 100 m offset while preserving exact recovery of `h`.

## Initial-condition family

The in-distribution family is a sum of one to three Gaussian surface perturbations:

$$
\eta(x,y,0) = \sum_{k=1}^{K} A_k
\exp\left[-\frac{(x-x_k)^2+(y-y_k)^2}{2\sigma_k^2}\right],
\quad K\in\{1,2,3\}.
$$

Default seeded ranges:

| Parameter | Distribution |
|---|---|
| Number of bumps `K` | discrete uniform over 1, 2, 3 |
| Amplitude `A_k` | signed uniform magnitude from 0.5 m to 1.5 m |
| Width `sigma_k` | uniform from 65 km to 120 km |
| Center `x_k`, `y_k` | uniform with a 2 `sigma_k` wall margin |
| Initial `u`, `v` | zero |

Reject and redraw a configuration if `H + eta <= 0`, if the perturbation violates
the wall margin, or if a quick stability preflight produces non-finite values. Record
`sigma/dx` for every grid as a resolvability diagnostic.

## Size and split

Create one immutable `ic_registry_v1.json` with 48 analytic IC specifications, seeds,
stable trajectory UUIDs, and split membership. Both resolution pairs consume it.
Generate 197 saved snapshots per trajectory (D017). The primary discards 288 steps and
saves every 24, ending at step 4992; the backup discards 576 and saves every 48, ending
at step 9984. Both the stride and the step cap are doubled for the backup because its
`dt` is half the primary's, which is what keeps the two pairs at matching physical times
and matching frame counts. Derive exact step counts and float64 physical times from
resolved configs, not rounded intervals.

| Split | Trajectories | Snapshot pairs | Seed IDs |
|---|---:|---:|---|
| Train | 32 | 6,304 | 0-31 |
| Validation | 8 | 1,576 | 32-39 |
| Test | 8 | 1,576 | 40-47 |

The split is fixed before simulation. A trajectory and all its time frames belong to
one split. Do not reshuffle snapshots across splits.

The combined float32 array payload is approximately 9.198 GiB before metadata and
compression. Backup solver work is roughly eight times primary work: about four times
the grid points and twice the time steps. Generate and validate primary first, then a
backup smoke dataset, then stream the full backup release.

## Training samples and augmentation

Use full-frame pairs: 32 -> 128 for the primary study and 64 -> 256 for a later
separate training run. Endpoint-aligned interpolation has an unambiguous physical
coordinate map; naive index-aligned patches do not.

- Use only symmetry-preserving augmentation: horizontal/vertical reflection with
  the sign of the corresponding velocity component changed.
- A transpose augmentation swaps axes and also swaps `u` with `v`; use it only after
  a unit test verifies the transformation.
- Do not use arbitrary image rotations, color transforms, independent channel
  scaling, or naive x4 index crops.

If patch training is later required, map patch bounds by physical coordinates and
record the method as a new decision. Full-frame states are always used for validation
and final evaluation.

## Normalization

Compute one mean and standard deviation per channel from fine-grid training states
only, separately for each resolution pair. Apply a pair's statistics to both its LR
and HR fields:

$$
q'_c = \frac{q_c-\mu_c}{\max(\sigma_c,10^{-8})}.
$$

Persist sample counts, sums, squared sums, `mu`, and `sigma` in the manifest. Never
share statistics across pair IDs or fit them on validation, test, or fresh data.

## Storage contract

Use chunked Zarr or HDF5 with this logical layout:

```text
data/
  registries/ic_registry_v1.json
  raw/<dataset_id>/
    trajectories/<trajectory_id>/{lr,hr,time,coordinates,metadata}
    manifest.json
  processed/<dataset_id>/
    manifest.json
    normalization.json
```

Each field array has shape `[time, channel, y, x]`. Each manifest records:

- dataset and schema version;
- Git commit and solver config hash;
- physical parameters and boundary conditions;
- grid coordinates, shared time step, and saved times;
- resolution family ID, pair ID, IC registry hash, endpoint convention, coordinate
  hashes, node counts, shape factor, physical spacing, and spacing ratio;
- initial-condition parameters and seed;
- split membership;
- dtype, shape, chunking, and per-array checksum;
- normalization statistics computed from the train split.

## Fresh workloads

Fresh workloads are generated only after model selection and are absent from the
training manifest.

1. `fresh_id`: a new seed from the Gaussian family, for a clean repeatability check.
2. `ring_ood`: an annular elevation perturbation with zero initial velocity,

$$
\eta(r,0)=A\exp[-(r-r_0)^2/(2\sigma^2)],
$$

   which is qualitatively different from the Gaussian training family.

Report the two separately. Do not mix their metrics into the held-out test score.
If generated at both resolutions, a fresh workload shares one evaluation-only ID.
Cross-resolution comparisons use paired trajectory bootstrapping because ICs match.
