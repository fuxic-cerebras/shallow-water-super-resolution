# Decision Log

## D001 - Three physical channels

- Status: accepted
- Decision: model `[eta, u, v]`; reconstruct total height as `h = H + eta`.
- Reason: `swe.py` evolves `eta`, and removing the constant depth offset improves
  numerical conditioning without discarding information.

## D002 - Independent coarse and fine integrations

- Status: accepted
- Decision: use independent 32/128 primary and 64/256 backup solver pairs over the
  same physical domain.
- Reason: the research question concerns error from a genuinely coarse PDE solve,
  not recovery from an artificially downsampled fine image.

## D003 - Shared fine-grid time step within each pair

- Status: accepted for version 1
- Decision: integrate LR and HR within each resolution pair with that pair's stable
  HR-grid time step. Different pair IDs may use different time steps.
- Reason: exact time alignment avoids interpolation and time-discretization mismatch.
  A later experiment may let each solver use its own stable time step.

## D004 - Trajectory-level split

- Status: accepted
- Decision: split 48 initial-condition trajectories into 32 train, 8 validation, and
  8 test trajectories before simulation.
- Reason: random frame splitting would leak nearly identical adjacent states.

## D005 - MSE-only primary study

- Status: accepted
- Decision: train on equal-weight normalized channel MSE.
- Reason: it directly tests the requested squared L2 objective and makes the U-Net
  versus EDSR comparison interpretable before adding physics losses.

## D006 - Common bicubic residual

- Status: accepted
- Decision: both networks learn a residual over bicubic x4 interpolation.
- Reason: both models start from the same low-frequency baseline, which improves
  optimization and keeps the comparison focused on learned detail.

## D007 - One-day data budget

- Status: accepted as a planning default
- Decision: the primary has 6,144 paired snapshots, approximately 1.2 GiB raw, and
  no more than 30,000 optimizer steps per model. Backup has a separate budget.
- Reason: this is a practical baseline that can be expanded after measured scaling.

## D008 - Dual resolution-pair releases

- Status: accepted
- Decision: release 32 -> 128 as primary and 64 -> 256 as backup, using one immutable
  IC/split registry but separate arrays, manifests, timesteps, normalization, and runs.
- Reason: preserve a fast first experiment while retaining higher-resolution data.

## D009 - Multi-agent ownership and independent verification

- Status: accepted
- Decision: use four specialist subagents plus an integration lead, path ownership,
  explicit handoffs, and independent verification gates.
- Reason: separate PDE/data from ML and prevent authors from self-certifying results.

## Template for new decisions

```text
## DNNN - Title

- Status: proposed | accepted | superseded
- Decision:
- Reason:
- Consequences:
```
