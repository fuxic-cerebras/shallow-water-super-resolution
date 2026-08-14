# Documentation index

Generated from `docs/index.yaml` by `python -m swe_sr.docgen render` (D024). Do not edit the
table below by hand — add the document to `docs/index.yaml`, or, if it is an experiment write-up,
drop it in `docs/experiments/` and it appears here on the next render.

`docs/DOCUMENTATION.md` explains the three kinds of document and why the distinction decides what
is allowed to grow.

## Where to start

If you are about to change code, read the contracts in order:

<!-- BEGIN generated: docs:reading-order -->
1. `docs/PROJECT_SPEC.md` — goal, scope, research questions, definition of done, open questions
2. `docs/DATASET.md` — resolution pairs, IC family, splits, normalization, storage contract
3. `docs/ARCHITECTURE.md` — repository layout, batch contract, the three model designs, run directory
4. `docs/EXPERIMENT_PLAN.md` — baselines, loss, optimizer schedule, staged runs, artifacts
5. `docs/VALIDATION.md` — data gates, metric definitions, aggregation protocol, negative tests
6. `TASKS.md` — task status and per-gate evidence
7. `docs/README.md` — the full index, including every experiment write-up
<!-- END generated: docs:reading-order -->

## Everything

<!-- BEGIN generated: docs:index -->
### Contracts

Normative. What must be true, with no results in them. A change to one of these is a decision record.

| File | Contents |
|---|---|
| [`PROJECT_SPEC.md`](PROJECT_SPEC.md) | goal, scope, research questions, definition of done, open questions |
| [`DATASET.md`](DATASET.md) | resolution pairs, IC family, splits, normalization, storage contract |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | repository layout, batch contract, the three model designs, run directory |
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | baselines, loss, optimizer schedule, staged runs, artifacts |
| [`VALIDATION.md`](VALIDATION.md) | data gates, metric definitions, aggregation protocol, negative tests |

### Guides

How the thing works and how to run it. Extracted from `README.md` so it stays a front door.

| File | Contents |
|---|---|
| [`PIPELINE.md`](PIPELINE.md) | the six stages, what each consumes and writes, and the rule behind each choice |
| [`REPRODUCE.md`](REPRODUCE.md) | the command-by-command rerun, measured timings, and what a rerun cannot reproduce |
| [`DOCUMENTATION.md`](DOCUMENTATION.md) | how documentation here is organized, and which numbers are generated rather than typed |
| [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md) | the multi-agent team, waves, gates, and the handoff lifecycle |

### Ledgers

Append-only, and expected to grow without bound. Nobody reads these front to back; they are searched by identifier.

| File | Contents |
|---|---|
| [`DECISIONS.md`](DECISIONS.md) | every project decision with its reason and consequences |
| [`EXPERIMENT_FREEZE.md`](EXPERIMENT_FREEZE.md) | the frozen T-03 record: hashes, commits, seeds, checkpoint digests |
| [`../TASKS.md`](../TASKS.md) | task status and per-gate evidence |
| [`SIGNOFF.md`](SIGNOFF.md) | the I-03 audit: what is verified, what is outstanding |

### Generated

Written by a tool from run artifacts. Never edited by hand.

| File | Contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | comparison table, lead-time breakdown, fresh workloads, limitations |
| [`results/index.json`](results/index.json) | the committed scalars every document's tables are rendered from |
| [`results/runs.yaml`](results/runs.yaml) | the run registry, and the only place a run ID is written by hand |

### References

What the design was taken from, and where it was checked.

| File | Contents |
|---|---|
| [`REFERENCES.md`](REFERENCES.md) | solver references and the three supplied papers, with an interpretation rule |
| [`RESEARCH_MATRIX.md`](RESEARCH_MATRIX.md) | reference-to-design matrix, verified against pinned upstream commits |

### Experiments

One file per experiment, immutable once its arm has landed. A new result is a new file, never an edit to an existing write-up. Globbed from `docs/experiments/`, so this list needs no maintenance.

|  | File | Contents |
|---|---|---|
| A-01 | [`experiments/A-01-residual-vs-direct.md`](experiments/A-01-residual-vs-direct.md) | Outer bicubic residual against direct prediction (D022) |
| A-03 | [`experiments/A-03-convmixer-normalization.md`](experiments/A-03-convmixer-normalization.md) | Does EDSR's "no normalization" advice transfer to ConvMixer? |
| A-05 | [`experiments/A-05-convmixer-regularization.md`](experiments/A-05-convmixer-regularization.md) | ConvMixer regularization, and the training-curve comparison |
| O-07 | [`experiments/O-07-cross-resolution-transfer.md`](experiments/O-07-cross-resolution-transfer.md) | Cross-resolution transfer: 32→128 models tested on 64→256 |
<!-- END generated: docs:index -->
