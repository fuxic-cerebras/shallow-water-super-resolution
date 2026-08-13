# References

## Solver and implementation references

All commits and licenses below were verified against the pinned sources on 2026-08-12
as part of R-01. The table is the authoritative pin list; prose sections add detail.

| Repository | Pinned commit | License | Role |
|---|---|---|---|
| `shallow-water` | `a8457df886cec74e2a02652280d2f00de0804dfc` | **none found** | solver, submodule at `references/shallow-water` |
| `space_time_pde` | `5e355b0434baf1757d071ce993b84073c8426223` | MIT (c) 2020 Chiyu Max Jiang, Soheil Esmaeilzadeh | U-Net / implicit-decoder evidence |
| `climatereconstructionAI` | `deb8582a8c390e4a72444f1787dce616f93d4d73` | BSD 3-Clause | training/validation workflow evidence |
| `EDSR-PyTorch` | `8dba5581a7502b92de9641eb431130d6c8ca5d7f` | MIT (c) 2018 Sanghyun Son | residual-block and pixel-shuffle evidence |

> **Licensing risk.** The pinned `shallow-water` fork contains no `LICENSE`, `COPYING`,
> or `NOTICE` file at its root, so its terms are unstated. This is why D010 tracks it
> as a submodule rather than vendoring the files: nothing from it is redistributed by
> this repository. Before any external release of derived solver code, confirm the
> upstream provenance and terms with the repository owner. Do not assume a license.

- [Shallow-water solver](https://github.com/fuxic-cerebras/shallow-water) - source
  `swe.py`, numerical scheme, physical parameters, and existing model notes. Tracked
  as a submodule at `references/shallow-water`; see D010. `MODEL_NOTES.md` lives here
  and is the source of truth for the equations and discretization.
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

### Witte et al. (2024)

"Dynamic Deep Learning Based Super-Resolution For The Shallow Water Equations,"
arXiv:2404.06400v2. Maximilian Witte, Fabricio Rodrigues Lapolli, Philip Freese,
Sebastian Götschel, Daniel Ruprecht, Peter Korn, Christopher Kadow.

> **Citation corrected 2026-08-13.** This entry previously read "Margenberg et al. (2024),
> *Dynamic Deep Learning Based Super-Resolution for Shallow Water Flows*". Both the author and
> the title were wrong; the arXiv ID was right, and every architectural claim below was
> re-checked against the source and holds. This is exactly the R-01 exception `TASKS.md`
> records — "the three source papers were not re-audited against their arXiv originals" — so
> treat the other two entries as carrying the same risk until they are checked the same way.

Relevant choices, verified against the arXiv v2 abstract and HTML full text on 2026-08-13
except where marked:

- corrects a coarse solution using a U-net-type network "trained to compute the difference
  between solutions on both meshes", applied to "correct the coarse mesh every 12h";
- two consecutive residual blocks per stage "with a swish activation function";
- "average pooling layers with a kernel size of 4" for downsampling;
- "sub-pixel convolutional blocks" with pixel shuffle for upsampling;
- encoder output "concatenated with ... upsampled output of the decoder layer";
- "convolutional layers of the U-net without biases", which is what gives the network the
  property `Theta(0) = 0`;
- runs high- and low-resolution shallow-water simulations from matching initial conditions;
- *not re-verified in the full text*: separate IC variants for training and validation,
  velocity normalization from training data, mini-batch training with a held-out simulation.

Three differences in this project, each deliberate:

- **Output.** The primary model here produces a true 128 x 128 field from a 32 x 32 input. The
  paper's network maps a coarse field to a *corrected coarse field* — a same-grid correction,
  later coupled into the solver at run time.
- **No interpolated-input skip in the paper.** It has no global additive path from an
  upsampled input, because it never produces a fine-grid field to add one to. This project's
  outer `y_hat = bicubic(x) + R(x)` is **D006, a project decision**, not an adoption from this
  paper. The nearest published lineage for adding an interpolated input is the VDSR family,
  which is not pinned here.
- **Bias-free convolutions.** The paper makes *all* U-net convolutions bias-free to guarantee
  `Theta(0) = 0`. `docs/ARCHITECTURE.md` specifies bias-free ends only, so interior
  convolutions here keep their biases and this project does **not** inherit that property. The
  analogous guarantee here comes from the outer residual instead: zeroed weights reproduce the
  bicubic baseline exactly. Average pooling is also kernel 2 here against the paper's 4.

## Interpretation rule

Paper settings are evidence, not defaults to copy blindly. Any adapted choice must
state what changed and why in `docs/ARCHITECTURE.md` or `docs/DECISIONS.md`.
