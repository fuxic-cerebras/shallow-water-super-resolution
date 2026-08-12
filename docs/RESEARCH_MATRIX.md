# Reference-to-Design Matrix

This file is the starting point for R-01. The Research agent verifies every row against
pinned source commits before G0.

| Reference | Useful evidence | Do not copy directly | Project adaptation |
|---|---|---|---|
| `shallow-water/swe.py` | Current PDE, boundaries, CFL form, `[eta,u,v]` fields | plotting/global-state execution path | headless tested solver with unchanged numerical kernel |
| Fukami et al. 1811.11328 | squared L2 objective, flow fields as channels, bicubic baseline | HR pooling as primary LR source; frame-random split | independent coarse solve, normalized per-channel MSE, trajectory split |
| `space_time_pde` / 2005.01463 | residual 3D U-Net, skip connections, implicit/PDE-loss extension | 3D baseline, HR-derived LR, whole-file statistics, legacy code | clean 2D x4 U-Net now; continuous/PDE loss only later |
| `climatereconstructionAI` | encoder/decoder organization, train-stat reuse, validation, checkpoints, early stopping | partial convolutions, masks, recurrence, same-grid infilling, heavy climate stack | borrow workflow patterns, implement direct x4 physical-field SR |
| 2404.06400 | matching HR/LR shallow-water simulations; residual blocks, Swish, average pooling, sub-pixel blocks | coarse-grid correction/coupling as if it were HR reconstruction | adapt architectural concepts to explicit x4 output |
| `EDSR-PyTorch` | residual blocks, residual scaling, pixel shuffle | RGB MeanShift, natural-image weights, old environment | three normalized physical channels and random initialization |

## Repository findings (verified 2026-08-12 against pinned commits)

- **Verified.** `space_time_pde/src/unet.py` is a same-size 2D U-Net
  (`in_channels=4, depth=5, up_mode='transpose'`) and its helper import *is* stale:
  line 8 is `from utils import *`, but at commit `5e355b0` the `src/` tree contains only
  `model_utils.py` and `train_utils.py` — no `utils.py`. The module cannot be imported
  as shipped. The paper path uses `src/unet3d.py` plus `src/implicit_net.py`, both of
  which do exist at that commit.
- **Verified.** Its data loader (`experiments/rb2d/dataloader_spacetime.py`) constructs
  LR from HR by downsampling and interpolating (`downsamp_xz=4`, `downsamp_t=4`,
  `lres_interp`, `scipy` `RegularGridInterpolator`), and computes channel-wise mean and
  std over the entire loaded file immediately after loading it, with no train-split
  restriction. Neither behavior satisfies this project's paired-solver contract (D002)
  or its train-only normalization rule.
- **Not independently verified.** `climatereconstructionAI` is taken on its
  documentation as masked climate infilling rather than x4 SR. Its intended
  contribution here is the surrounding reproducible training and validation workflow,
  and no code from it is on the critical path, so a deeper audit was deferred rather
  than performed.
- Both reference codebases require adaptation and dependency modernization. Do not
  vendor either repository wholesale.

## G0 output

Pinned commits and verified licenses for all four repositories are tabulated in
`docs/REFERENCES.md`, including the finding that the `shallow-water` fork carries no
license file. Copied-versus-reimplemented decisions: nothing is copied. The solver is
transcribed from a pinned submodule under regression test (D010); the U-Net and EDSR
are reimplemented from documented architectural evidence, not adapted from source.

Unresolved conflicts handed to I-01, all now resolved in `docs/DECISIONS.md`: the
C-grid staggering versus colocated-channel storage conflict (D011), the undefined
discrete energy convention (D014), and the absent CUDA device assumed by the runtime
budget (D015). Three discretization ambiguities that G0 could not resolve are recorded
in the Open Questions section of `docs/PROJECT_SPEC.md`.
