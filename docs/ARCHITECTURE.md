# Architecture

## Proposed repository layout

```text
references/
  shallow-water/        # submodule, pinned at a8457df, read-only
configs/
  data/{primary_32x128,backup_64x256}.yaml
  model/{unet_x4,edsr_x4}.yaml
  experiment/{smoke,pilot,full}.yaml
swe_sr/
  solver/{config,initial_conditions,numerics,runner}.py
  data/{generate,dataset,manifest,normalization,validate}.py
  models/{unet,edsr,common}.py
  metrics/{field,physics}.py
  train.py
  evaluate.py
  evaluate_fresh.py
tests/
  solver/
  data/
  models/
  metrics/
docs/
runs/                 # ignored by Git
data/                 # ignored by Git
```

The reference `references/shallow-water/swe.py` remains an untouched demonstration
script inside its pinned submodule (D010); it is not importable from the repository
root and is never edited. Numerical kernels move behind the importable `swe_sr.solver`
API only after regression tests capture current behavior, with the fixtures produced by
executing the submodule script under patched globals.

## Shared input/output contract

- Input shape: `[batch, 3, H, W]` in `[eta, u, v]` order.
- Output shape: `[batch, 3, 4H, 4W]` in the same order.
- Required smoke shapes are 32 -> 128 and 64 -> 256.
- Input and target use training-split channel normalization.
- Both models predict a residual added to bicubic interpolation of the input.
- Bicubic interpolation aligns the physical domain endpoints; its exact library
  options are fixed in config and shared by both models and the baseline.
- No RGB mean-shift layer, clipping to image ranges, or pretrained natural-image
  weights are used.
- Output is unconstrained during training. Physical diagnostics are computed after
  de-normalization.

The common resolution-generic residual formulation is

$$
\hat{y}=\operatorname{bicubic}(x)+R_\theta(x).
$$

## Residual U-Net x4

Adapt the multi-scale ideas in the dynamic shallow-water U-Net while keeping the model
small enough for a one-day study:

1. Bicubic-upsample the normalized input to its x4 output shape.
2. Use a bias-free 3 x 3 input convolution with 32 features.
3. Use three encoder stages with 32, 64, and 128 features.
4. Each stage has two residual blocks with SiLU/Swish activation.
5. Downsample with 2 x 2 average pooling.
6. Decode with pixel-shuffle x2 blocks and concatenate matching encoder features.
7. Use a bias-free 3 x 3 output convolution to predict three residual channels.

This preserves the U-Net skip connections, residual blocks, average pooling, Swish,
and sub-pixel decoder concepts used in the 2024 dynamic-super-resolution paper, but
adapts them from same-grid correction to true x4 image-to-image reconstruction.

Use the repositories as evidence, not vendored implementations:

- `space_time_pde/src/unet3d.py` supplies residual U-Net patterns, but the paper path
  is 3D space-time plus an implicit decoder and derives LR from HR.
- `climatereconstructionAI/climatereconstructionai/model/` supplies maintained
  encoder/decoder and training patterns, but partial convolutions, masks, recurrence,
  and same-resolution infilling are out of scope.
- The baseline remains a clean 2D residual U-Net with explicit x4 output. Pin source
  commit SHAs and licenses before adapting any code.

## EDSR x4

Adapt the baseline configuration from `fuxic-cerebras/EDSR-PyTorch`:

- 16 residual blocks;
- 64 feature channels;
- 3 x 3 convolutions;
- ReLU activation;
- residual scaling of 0.1;
- global body skip connection;
- two pixel-shuffle x2 blocks for x4 output;
- three physical input and output channels;
- an outer residual added to bicubic interpolation for parity with U-Net.

Do not use the repository's RGB `MeanShift`; dataset normalization replaces it. Start
from random initialization because natural-image weights do not share field semantics.

## Why both models

| Model | Expected strength | Main risk |
|---|---|---|
| U-Net | Multi-scale context and direct high-resolution refinement | Higher memory and compute at 128 x 128 |
| EDSR | Efficient residual feature extraction mostly on the coarse grid | May miss basin-scale context or over-smooth coupled fields |

The comparison is empirical. Parameter count, peak memory, throughput, and inference
latency must be reported alongside accuracy.

## Solver API boundary

The refactored solver should expose pure data, not plotting side effects:

```python
state = solve(config, initial_condition, sample_steps)
# state.fields.shape == [time, 3, ny, nx]
# state.fields[:, 0] is eta; metadata contains H and coordinates
```

`viz_tools.py`, GIF generation, `plt.show()`, and printing are optional clients. Data
generation must run headlessly and must not accumulate every internal time step.

## Configuration and provenance

Every run directory contains:

```text
runs/<run_id>/
  config.yaml
  environment.json
  dataset_manifest.json
  metrics.csv
  summary.json
  curves.png
  checkpoints/{best,last}.pt
  predictions/          # optional, bounded examples only
```

The run ID includes timestamp, model name, short config hash, and Git commit. A run
must be reproducible without relying on mutable global constants in `swe.py`.
