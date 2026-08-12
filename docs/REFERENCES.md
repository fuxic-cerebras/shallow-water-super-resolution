# References

## Solver and implementation references

- [Shallow-water solver](https://github.com/fuxic-cerebras/shallow-water) - source
  `swe.py`, numerical scheme, physical parameters, and existing model notes.
- [EDSR-PyTorch fork](https://github.com/fuxic-cerebras/EDSR-PyTorch) - reference
  implementation for residual blocks and pixel-shuffle x4 upsampling. The project
  adapts it from RGB images to normalized physical channels and does not reuse RGB
  mean shift or natural-image weights.
- [MeshfreeFlowNet implementation](https://github.com/fuxic-cerebras/space_time_pde) -
  U-Net, implicit decoding, and PDE-loss reference for arXiv:2005.01463. Pin commit
  `5e355b0434baf1757d071ce993b84073c8426223` (MIT); do not copy its HR-derived LR path.
- [Climate Reconstruction AI](https://github.com/fuxic-cerebras/climatereconstructionAI) -
  U-Net ecosystem and validation/checkpointing reference associated with the dynamic
  SR work. Pin commit `deb8582a8c390e4a72444f1787dce616f93d4d73` (BSD-3-Clause).
  Partial-convolution masked infilling is not direct x4 SR.

## Papers supplied with this project

### Fukami, Fukagata, and Taira (2019)

"Super-resolution reconstruction of turbulent flows with machine learning,"
arXiv:1811.11328.

Relevant choices:

- formulates reconstruction as squared L2 minimization;
- treats vector flow fields as image channels;
- splits snapshots 70/30 for training/validation and tests on excluded snapshots;
- compares with bicubic interpolation and reports normalized L2 error;
- shows that snapshot count and the low-resolution data-generation process matter.

Difference in this project: low-resolution inputs come from an independent coarse
solver, not pooling of high-resolution snapshots, and splits are by trajectory.

### Jiang et al. (2020)

"MeshfreeFlowNet: A Physics-Constrained Deep Continuous Space-Time
Super-Resolution Framework," arXiv:2005.01463.

Relevant choices:

- represents physical quantities as multiple channels;
- uses a residual 3D U-Net context network;
- evaluates generalization to unseen initial and boundary conditions;
- demonstrates that training on diverse initial conditions improves unseen-case
  performance;
- combines data and PDE losses, while also reporting a data-loss-only baseline.

Difference in this project: version 1 is discrete 2D spatial SR at one time, uses an
MSE-only baseline, and defers continuous space-time decoding and PDE loss.

### Margenberg et al. (2024)

"Dynamic Deep Learning Based Super-Resolution for Shallow Water Flows,"
arXiv:2404.06400v2.

Relevant choices:

- runs high- and low-resolution shallow-water simulations from matching initial
  conditions;
- uses separate initial-condition variants for training and validation;
- normalizes velocity components from training data;
- uses a U-Net with skip connections, residual blocks, Swish activations, average
  pooling, and sub-pixel convolution;
- trains with mini-batches and monitors a held-out simulation;
- tests a new runtime coupling rather than only training snapshots.

Difference in this project: the primary model produces a true 128 x 128 field from a
32 x 32 input. The paper instead corrects a coarse state toward the fine solution
restricted back to the coarse grid and later couples that correction into the solver.

## Interpretation rule

Paper settings are evidence, not defaults to copy blindly. Any adapted choice must
state what changed and why in `docs/ARCHITECTURE.md` or `docs/DECISIONS.md`.
