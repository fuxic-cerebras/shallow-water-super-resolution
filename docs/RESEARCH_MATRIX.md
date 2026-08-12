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

## Repository findings to verify

- `space_time_pde/src/unet.py` is a same-size 2D U-Net and appears to have a stale
  helper import. The paper experiment uses `src/unet3d.py` plus an implicit decoder.
- Its data loader constructs LR from HR and calculates statistics over the loaded file;
  neither behavior satisfies this project's paired-solver or train-only-stat contract.
- `climatereconstructionAI` implements masked climate infilling, not x4 SR. Its useful
  contribution here is the surrounding reproducible training and validation workflow.
- Both reference codebases require adaptation and dependency modernization. Do not
  vendor either repository wholesale.

## G0 output

R-01 adds exact file links, licenses, pinned commits for all repositories, copied versus
reimplemented code decisions, and unresolved conflicts. I-01 freezes the chosen design.
