# Shallow-Water Super-Resolution

This repository develops neural spatial super-resolution for the 2D shallow-water
solver in `references/shallow-water/swe.py`. A coarse numerical solution is the model
input; an independently integrated fine-grid solution at the same physical time is the
target.

## Read first

Before changing code, read these files in order:

1. `docs/PROJECT_SPEC.md`
2. `docs/DATASET.md`
3. `docs/ARCHITECTURE.md`
4. `docs/EXPERIMENT_PLAN.md`
5. `docs/VALIDATION.md`
6. `docs/AGENT_WORKFLOW.md`
7. `docs/RESEARCH_MATRIX.md`
8. `TASKS.md`

`references/shallow-water/MODEL_NOTES.md` is the source of truth for the existing
solver's equations and discretization. It lives in a submodule pinned at commit
`a8457df`; run `git submodule update --init` if the directory is empty.
`docs/DECISIONS.md` records project-level choices.

## Non-negotiable rules

- Treat the solver output as three physical channels ordered `[eta, u, v]`.
  The total height is `h = H + eta`; do not silently rename `eta` to `h`.
- Generate low-resolution and high-resolution states by running two solver
  discretizations. Do not create the primary low-resolution dataset by resizing
  high-resolution arrays.
- Use the same physical domain, parameters, analytic initial condition, time step,
  and output times for each paired run. Only spatial resolution changes.
- Split by trajectory and initial-condition seed, never by individual snapshots.
  Adjacent time frames from one trajectory must not cross a split boundary.
- Fit normalization statistics on the training split only. Persist the statistics
  in the dataset manifest and use the same values for validation, test, and inference.
- Keep physical units in stored raw data. Normalize only in the data loader.
- Never report a metric without its units, aggregation rule, split, and baseline.
- Never invent experiment results. Mark planned numbers as targets and measured
  numbers as results with the command, config, seed, and checkpoint that produced them.
- Do not commit generated datasets, model checkpoints, or experiment logs to Git.
- Preserve the original `references/shallow-water/swe.py` behavior while extracting
  reusable solver code. Add regression tests before changing numerical kernels. The
  submodule is read-only; never edit it.
- Treat `[eta, u, v]` as an Arakawa C-grid in the raw layer and as cell-centered
  fields in the processed layer, per D011. Models and metrics consume processed,
  cell-centered fields only.
- Record any change to the grid ratio, fields, split seeds, loss definition, or
  evaluation protocol in `docs/DECISIONS.md`.
- Treat 32 -> 128 as the primary dataset and 64 -> 256 as a separately generated
  backup dataset. They share initial-condition identities and splits, not manifests,
  normalization statistics, checkpoints, or training runs.

## Multi-agent workflow

Follow `docs/AGENT_WORKFLOW.md`. The default team is four specialist subagents plus
the integration lead: Research and Design, PDE and Data, ML and Training, Testing and
Scientific Verification, and the Integration Lead.

An agent must claim a task ID, stay within owned paths, and produce a handoff packet.
Authors set work to `ready-for-review`; only the independent verifier and integration
lead may mark it complete. Do not edit shared contracts or run full dataset/training
jobs concurrently without approval from the integration lead.

## Working style

- Work on one unchecked item in `TASKS.md` at a time.
- Start each task by stating the acceptance criterion and the files to change.
- Prefer small, reviewable modules and configuration-driven experiments.
- Add or update tests in the same change as behavior.
- Run the narrowest relevant tests first, then the full fast test suite.
- Update documentation and task status only after validation passes.
- At every major milestone, once its verification actually passes, commit the change and
  merge to `main`. Never accumulate a gate's worth of work into one large final commit.
  A milestone is a gate (G0-G8) or a completed `TASKS.md` task ID. Run the relevant tests
  first; if part of a gate fails, commit the passing part and state what is outstanding.
- If an assumption is unclear, stop and add it to the Open Questions section of
  `docs/PROJECT_SPEC.md` instead of burying it in code.

## Expected commands

The implementation should converge on these stable entry points:

```bash
python -m swe_sr.data.generate --config configs/data/primary_32x128.yaml
python -m swe_sr.data.validate --manifest data/processed/swe_gaussian_32x128_v1/manifest.json
python -m swe_sr.train --config configs/model/unet_x4.yaml
python -m swe_sr.train --config configs/model/edsr_x4.yaml
python -m swe_sr.evaluate --run-dir runs/<run-id>
python -m swe_sr.evaluate_fresh --scenario ring_ood --run-dir runs/<run-id>
pytest -q
```

If a command is not implemented yet, implement it through the corresponding task;
do not add an undocumented alternate entry point.
