# Run a reproducible experiment

Read `CLAUDE.md`, `docs/EXPERIMENT_PLAN.md`, and `docs/VALIDATION.md`.

Given the requested experiment config:

1. Validate the dataset manifest and confirm split IDs.
2. Print the resolved config, config hash, Git commit, seed, and device.
3. Refuse to overwrite an existing run directory.
4. Run the experiment and preserve stdout/stderr in the run directory.
5. Evaluate the best-validation checkpoint, not an arbitrary last checkpoint.
6. Generate curves and machine-readable metrics.
7. Report actual wall time, peak memory, and any deviation from the requested config.

Label planned values as targets and measured values as results. Never compare methods
that used different test manifests without explicitly marking the comparison invalid.
