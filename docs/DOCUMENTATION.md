# How documentation in this repository works

The problem this solves is not that the documentation was long. It was that the same number
lived in many files. Before D024 the headline `0.0400` appeared in eight of them and `0.0830` in
nine, each typed by hand, so re-running one arm meant editing nine files consistently and
nothing detected a missed edit. Drift had already begun: `README.md` described the decision log
as running "D001 to D021" when D023 existed, and its own index omitted two write-ups its prose
linked to.

So the rule is not "write less". It is **write once, in the place whose lifetime matches what
you are writing**, and let anything derived be generated.

## Three kinds of document

Which kind a file is determines whether growth is a problem.

| Kind | Files | Changes | Growth |
|---|---|---|---|
| **Contract** | `PROJECT_SPEC`, `DATASET`, `ARCHITECTURE`, `EXPERIMENT_PLAN`, `VALIDATION` | rarely, and a change is a decision | a smell — keep under ~200 lines |
| **Ledger** | `DECISIONS.md`, `TASKS.md`, `EXPERIMENT_FREEZE.md` | append only | expected and fine |
| **Write-up** | `docs/experiments/*.md` | never, once its arm has landed | one new file per experiment |

**Contracts** state what must be true. They are normative and they carry **no results** — a
measured number in a contract is how a contract silently becomes a report. If a contract needs
to grow a paragraph of justification, the justification is a decision entry and the contract
keeps one line plus `(D0NN)`.

**Ledgers** are supposed to grow without bound. `DECISIONS.md` at 600 lines is not a problem,
because nobody reads it front to back; they search for a `DNNN`. The only requirements are that
each entry is bounded and that the file is reachable. Split a ledger into one file per entry
only when entries stop being uniform enough to skim.

**Write-ups** are where growth actually lands, because every new arm adds one. That is why they
are immutable: a new result is a **new file**, never an edit to an existing write-up. An
experiment that has been superseded says so in a line at the top and keeps its numbers, exactly
as a freeze record does. Name them `<ID>-<slug>.md` where the ID is a `TASKS.md` task (`A-05`) or
a decision (`D022`); the index reads the identifier from the filename and rejects one that has
none.

`docs/index.yaml` also carries two supporting sections — **guides** (`PIPELINE`, `REPRODUCE`,
this file, `AGENT_WORKFLOW`) and **references** (`REFERENCES`, `RESEARCH_MATRIX`) — plus the
**generated** files. None of those three grow with the number of experiments, which is why they
need no rule beyond staying accurate.

## Adding an experiment

```bash
# 1. run it, evaluate it
python -m swe_sr.train    --config configs/model/<arm>.yaml --experiment configs/experiment/full.yaml
python -m swe_sr.evaluate --run-dir runs/<run-id>

# 2. register the run -- the ONLY place a run ID is typed
$EDITOR docs/results/runs.yaml

# 3. promote its numbers into the committed index
python -m swe_sr.results --write

# 4. write up the finding in docs/experiments/<ID>-<slug>.md, with tables as empty blocks
# 5. render them
python -m swe_sr.docgen render

# 6. append a TASKS.md evidence entry and, if a contract changed, a DECISIONS.md entry
./scripts/check.sh
```

Step 4 is the one that takes judgment; steps 3 and 5 are mechanical and must not be done by
hand. If you find yourself typing a metric into prose, stop: either it belongs in a block, or
the index is missing a field.

## Blocks

A table in a document is delimited, and its contents come from `docs/results/index.json`:

```markdown
<!-- BEGIN generated: results:headline -->
<!-- END generated: results:headline -->
```

`python -m swe_sr.docgen render` fills it. `check` fails if it is stale. Renderers live in
`BLOCKS` in `swe_sr/docgen.py`; a renderer with no block in any document is an error, because an
unplaced renderer is untested.

`verified` blocks are the exception:

```markdown
<!-- BEGIN verified: results:frozen-test -->
<!-- END verified: results:frozen-test -->
```

These are checked and **never** rewritten. `docs/EXPERIMENT_FREEZE.md`'s results table is one:
rewriting a freeze to agree with whatever the artifacts currently say would destroy the only
property a freeze has. A mismatch there means a new decision and a new freeze.

`docs/RESULTS.md` is a third case — a wholly generated file, written by `swe_sr.report` from run
artifacts. It is not drift-checked, because it reports `ms/frame` and so is not byte-stable
across hosts. Cite the index, not `RESULTS.md`, when a number needs to appear elsewhere.

## What the checks enforce

`scripts/check.sh` runs `python -m swe_sr.docgen check`, which fails on:

- a generated block that no longer matches the index, or a verified block that no longer matches
  the artifacts;
- a backticked repository path that does not resolve — so a renamed document cannot leave
  live-looking pointers behind. Paths rooted outside this repository are skipped, because
  `docs/RESEARCH_MATRIX.md` correctly cites upstream files that do not exist here;
- a document under `docs/` that `docs/index.yaml` does not account for, or an index entry whose
  file does not exist. This is the rule that fixes what started all of it: the old hand-written
  table in `README.md` had silently omitted two write-ups its own prose linked to;
- a `DNNN` citation with no entry in `docs/DECISIONS.md`;
- a run ID cited in prose but absent from `docs/results/runs.yaml`.

None of this needs `runs/`, which is gitignored. That is the reason `docs/results/index.json` is
committed: it is the ~150 KB of scalars the prose cites, and committing it is what lets a clone
with no artifacts verify every table.

## When a document gets too long

Ask which kind it is.

- A **contract** over ~200 lines is usually carrying results or a tutorial. Move them out.
- A **ledger** is fine at any length. Do not "tidy" it by deleting entries; a superseded
  decision is evidence, and `TASKS.md` history is how a gate is audited.
- A **write-up** should not grow at all after its arm lands. If you are adding to one, the
  addition is a new experiment.
- `README.md` is a front door, not a manual: what the project is, the headline result, how to
  start, the command reference, and pointers. The pipeline tutorial is `docs/PIPELINE.md` and
  the from-scratch reproduction is `docs/REPRODUCE.md` for exactly this reason.

And if the same sentence is true in two files, one of them should link instead.
