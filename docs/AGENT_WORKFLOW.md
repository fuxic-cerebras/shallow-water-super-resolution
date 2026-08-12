# Multi-Agent Workflow

## Team

Use four specialist subagents plus the Integration Lead. Three agents are workable for
a prototype, but combining PDE/data and ML becomes a bottleneck and weakens review.

| Role | Owns | Must not do |
|---|---|---|
| Research and Design | paper/repository evidence, reference matrix, decisions proposed to lead | edit production code or claim measured results |
| PDE and Data | `swe_sr/solver`, `swe_sr/data`, data configs/tests | change model/loss contracts unilaterally |
| ML and Training | `swe_sr/models`, training code/configs/tests | alter manifests or solver numerics |
| Scientific Verifier | independent tests, audits, verification reports | silently repair code being judged |
| Integration Lead | contracts, tasks, decisions, integration, freezes, sign-off | waive failed gates without a decision record |

Specialists run concurrently only on disjoint paths. Shared schemas/docs, manifests,
full dataset generation, and accelerator training are serialized by the lead.

## Waves

1. Research freezes the evidence matrix, source commits/licenses, and adaptations.
2. PDE/Data and ML work in parallel against a frozen batch/model contract; ML uses
   synthetic fixtures until real data pass G3.
3. Verification prepares independent oracles and negative tests, then audits handoffs.
4. The lead integrates, freezes artifacts, schedules heavy compute, and signs off.

## Gates

| Gate | Required evidence |
|---|---|
| G0 Research freeze | Evidence matrix, commits/licenses, adaptation deltas, open conflicts |
| G1 Solver parity | Deterministic regression, headless import, physical diagnostics |
| G2 Data smoke | Both pairs independently integrated, aligned time/domain, valid arrays |
| G3 Dataset release | Disjoint splits, recomputed train stats/checksums, immutable manifest |
| G4 Model readiness | `[B,3,H,W] -> [B,3,4H,4W]` for H=32 and 64, gradients, reload |
| G5 Training readiness | Smoke/pilot, validation, curves, runtime/memory projection |
| G6 Experiment freeze | Manifests, commit, configs, stats, seed, metrics, checkpoint rule frozen |
| G7 Independent verification | Verifier reruns tests, data audit, baselines, evaluation |
| G8 Final review | Clean install, reproducible commands, docs/results match, lead sign-off |

Primary data pass G3 before full training. Backup data are a separate G3 release and
must not delay the primary experiment unless the lead records the change.

## Lifecycle and handoff

`unclaimed -> in-progress -> ready-for-review -> verified -> complete`

Authors stop at `ready-for-review`; the verifier records `verified`, and only the lead
records `complete`. One writer owns a path at a time.

Every handoff contains task ID/owner, branch or change identifier, exact files and
commands, measured results, artifact IDs/paths, deviations, risks/failures, next owner,
and requested gate. Never overwrite a dataset or run; use immutable unique IDs.
