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

## ConvMixer x4

Adapt `ConvMixer-256/16` from Trockman & Kolter, "Patches Are All You Need?"
(arXiv:2201.09792), per D023:

- patch-embedding stem, `patch_size` 1, 256 features;
- 16 repetitions of the ConvMixer block, at constant resolution throughout;
- each block is a **9 x 9 depthwise** convolution with a residual, then a **pointwise
  1 x 1** convolution, each followed by GELU and BatchNorm;
- the residual wraps the depthwise convolution only, as published;
- 1 x 1 projection to 64 channels;
- two pixel-shuffle x2 blocks for x4 output;
- three physical input and output channels;
- an outer residual added to bicubic interpolation for parity with the other two models.

The single residual placement is measured rather than incidental: adding a second one
around the pointwise convolution costs 1.10% on CIFAR-10 (paper Table 3).

`k=9` and `p=1` come from the paper's **CIFAR-10** ablations rather than its ImageNet
ones, because those are run at 32 x 32 — exactly this project's low resolution. `k=9`
is the knee of the kernel sweep (3 -> 93.61%, 5 -> 95.11%, 7 -> 95.72%, 9 -> 95.88%,
and only +0.28% beyond), and `p=1` is that sweep's preference at this input size
(p=2 costs 0.80%, p=4 costs 3.27%). Patching would discard the high-frequency content
super-resolution exists to recover, so the stem is a pointwise lift and all upsampling
happens in the decoder.

Two departures are structural. The classifier head (global average pooling plus a
linear layer) would destroy the spatial field, so it is replaced by EDSR's decoder. The
1 x 1 projection before that decoder is required for capacity reasons:
`PixelShuffleUpsampler(features)` costs `36h^2 + 4h`, so a full-width 256 decoder would
be 4.7 M parameters, larger than the entire body. Projecting first makes the decoder
byte-identical to EDSR's, so the two differ in body and not decoder.

Unlike the other two models, this one **keeps BatchNorm**, as published. That
contradicts EDSR's finding and is deliberate; D023 records the reasoning and the
eval-mode caveat.

## Why these three models

| Model | Receptive field | Expected strength | Main risk |
|---|---|---|---|
| U-Net | Global, via pooling | Multi-scale context and direct high-resolution refinement | Higher memory and compute at 128 x 128 |
| EDSR | ~33 px | Efficient residual feature extraction mostly on the coarse grid | May miss basin-scale context or over-smooth coupled fields |
| ConvMixer | 129 px, no pooling | Basin-scale context at constant resolution, so no pyramid to lose detail through | Large-kernel depthwise convolution is slow on CPU; BatchNorm is a known risk in super-resolution |

The three separate distinct inductive biases at comparable capacity — pyramid,
stacked-small-kernel, and isotropic-large-kernel — rather than three variations on one.
ConvMixer's 129 px exceeds both required input grids (32 and 64), so one unit integrates
the entire basin without any downsampling; EDSR cannot reach basin scale at all, and the
U-Net only reaches it by pooling.

Parameter counts are 1,517,571 (EDSR), 1,720,067 (ConvMixer), and 1,930,208 (U-Net), so
the comparison is capacity-ordered and no result can be attributed to size alone.

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
