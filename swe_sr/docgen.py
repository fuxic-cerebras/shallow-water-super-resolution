"""Render the documentation's tables from `docs/results/index.json`, and check for drift (D024).

    python -m swe_sr.docgen render     # rewrite generated blocks in place
    python -m swe_sr.docgen check      # fail on drift; rewrite nothing (this is what CI runs)
    python -m swe_sr.docgen lint       # hygiene only: links, orphans, decision refs, run IDs

A number gets into a document by being rendered here, never by being typed. Before this module
`0.0400` was transcribed into eight files and `0.0830` into nine, so re-running an arm meant
editing every one consistently and nothing detected a missed edit. `check` runs in
`scripts/check.sh`, so a stale table fails CI instead of rotting.

Two kinds of block, distinguished by their opening marker:

    <!-- BEGIN generated: results:headline -->      rewritten by `render`
    <!-- END generated: results:headline -->

    <!-- BEGIN verified: results:frozen-test -->    never rewritten, only checked
    <!-- END verified: results:frozen-test -->

The distinction exists for `docs/EXPERIMENT_FREEZE.md`. A freeze record is a claim that
specific numbers were obtained and will not move; silently rewriting it to match whatever the
artifacts now say would destroy exactly the property it exists to provide. So a `verified`
block is compared and never touched, and a mismatch is an error demanding a new freeze rather
than an edit. `render` reports one and leaves the file alone.

Everything renders from the committed index rather than from `runs/`, which is gitignored, so
`check` works in a fresh clone with no artifacts present.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from swe_sr.results import INDEX_PATH, REPO_ROOT, load_registry

DOCS_ROOT = REPO_ROOT / "docs"
DECISION_LOG = DOCS_ROOT / "DECISIONS.md"
DOC_REGISTRY = DOCS_ROOT / "index.yaml"

BEGIN = re.compile(r"^<!-- BEGIN (generated|verified): ([a-z0-9:_-]+) -->$")
END = re.compile(r"^<!-- END (generated|verified): ([a-z0-9:_-]+) -->$")

RUN_ID = re.compile(r"\b\d{8}T\d{6}Z_[a-z0-9_]+_[0-9a-f]{8}_[0-9a-f]{8}\b")
DECISION_REF = re.compile(r"\bD0\d{2}\b")
DECISION_HEADING = re.compile(r"^## (D0\d{2}) ", re.MULTILINE)
# Backticked paths only, and only ones containing a separator: a bare `swe.py` is ambiguous
# between a filename and a module, while `references/shallow-water/swe.py` is checkable. The
# character class excludes `<` and `*`, which is what keeps placeholders like `runs/<run-id>`
# and globs like `evaluation_*.json` out of the link check rather than failing on them.
BACKTICKED_PATH = re.compile(r"`([\w.-]+(?:/[\w.-]+)+\.(?:md|py|ya?ml|json|sh|sbatch|txt))`")

EXPERIMENT_FILENAME = re.compile(r"^([A-Z]-\d{2}|D\d{3})-[a-z0-9-]+$")

BASELINE_ORDER = ("nearest", "bicubic")


class DriftError(RuntimeError):
    """A document does not match what the index renders."""


# --------------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------------


def _display(path: Path) -> str:
    """Repository-relative where possible, so messages are short and clickable.

    Falls back to the full path for a file outside the repository, which is how the lint rules
    are exercised against fixtures rather than only against a repository that already passes.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _f4(value: float) -> str:
    return f"{value:.4f}"


def _ci(block: dict[str, Any], fmt: str = ".4f") -> str:
    return f"[{block['ci_low']:{fmt}}, {block['ci_high']:{fmt}}]"


def _bold(text: str, *, when: bool) -> str:
    return f"**{text}**" if when else text


def _table(header: Iterable[str], alignments: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(alignments) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _arms(index: dict[str, Any], predicate: Callable[[str, dict[str, Any]], bool]) -> list[str]:
    """Arm ids matching a predicate, ordered best test score first."""
    selected = [name for name, arm in index["arms"].items() if predicate(name, arm)]
    return sorted(selected, key=lambda name: index["arms"][name]["test"]["normmse"]["mean"])


# --------------------------------------------------------------------------------------------
# block renderers
# --------------------------------------------------------------------------------------------


def _headline_rows(
    index: dict[str, Any], arm_ids: list[str], columns: list[str], *, label_key: str = "label"
) -> tuple[list[list[str]], dict[str, float]]:
    """Rows for a method-comparison table, plus the per-column minimum to bold.

    The minimum is taken over baselines as well as models. That is the point of bolding here:
    U-Net's mass error is the best in the table only because it also beats bicubic's, and a
    model that lost to bicubic on a column should not be bolded for winning among models.
    """
    values: list[tuple[str, dict[str, Any]]] = [
        (name, index["baselines"][name]) for name in BASELINE_ORDER if name in index["baselines"]
    ]
    values += [(index["arms"][name][label_key], index["arms"][name]["test"]) for name in arm_ids]

    def column(block: dict[str, Any], key: str) -> float:
        if key == "normmse":
            return float(block["normmse"]["mean"])
        if key.startswith("rel_l2_"):
            return float(block["rel_l2"][key.removeprefix("rel_l2_")])
        if key == "mass":
            return float(block["relative_mass_error"])
        raise KeyError(key)

    minima = {key: min(column(block, key) for _, block in values) for key in columns}
    rows: list[list[str]] = []
    for label, block in values:
        row = [label, f"{block['params']:,}"]
        for key in columns:
            value = column(block, key)
            row.append(_bold(_f4(value), when=value == minima[key]))
            if key == "normmse":
                row.append(_ci(block["normmse"]))
        rows.append(row)
    return rows, minima


def block_headline(index: dict[str, Any]) -> str:
    """The five-row summary: baselines, then models worst to best so the winner ends the table."""
    arm_ids = _arms(index, lambda _, arm: arm["role"] == "primary")[::-1]
    rows, _ = _headline_rows(index, arm_ids, ["normmse"])
    return _table(["Method", "Params", "normMSE", "95% CI"], ["---", "---:", "---:", "---"], rows)


def block_frozen_test(index: dict[str, Any]) -> str:
    """The frozen T-03 table. Verified, never rewritten -- see this module's docstring."""
    arm_ids = _arms(index, lambda _, arm: arm["frozen"])[::-1]
    columns = ["normmse", "rel_l2_eta", "rel_l2_u", "rel_l2_v", "mass"]
    rows, _ = _headline_rows(index, arm_ids, columns, label_key="short")
    return _table(
        ["Method", "Params", "normMSE", "95% CI", "eta relL2", "u relL2", "v relL2", "mass err"],
        ["---", "---:", "---:", "---", "---:", "---:", "---:", "---:"],
        rows,
    )


def _training_rows(index: dict[str, Any], arm_ids: list[str], *, labels: dict[str, str]) -> str:
    best_test = min(index["arms"][name]["test"]["normmse"]["mean"] for name in arm_ids)
    best_val = min(index["arms"][name]["training"]["best_validation_mse"] for name in arm_ids)
    rows = []
    for name in arm_ids:
        arm = index["arms"][name]
        training = arm["training"]
        test = arm["test"]["normmse"]
        rows.append(
            [
                labels[name],
                f"{arm['test']['params']:,}",
                _bold(
                    _f4(training["best_validation_mse"]),
                    when=training["best_validation_mse"] == best_val,
                ),
                str(training["best_epoch"]),
                str(training["epochs"]),
                _f4(training["final_train_mse"]),
                f"{training['gap']:.2f}x",
                _bold(_f4(test["mean"]), when=test["mean"] == best_test),
                _ci(test),
            ]
        )
    return _table(
        [
            "model",
            "params",
            "best val",
            "best ep",
            "epochs",
            "final train",
            "gap",
            "test",
            "95% CI",
        ],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        rows,
    )


def block_architectures(index: dict[str, Any]) -> str:
    arm_ids = _arms(index, lambda _, arm: arm["role"] == "primary")
    labels = {
        name: index["arms"][name].get("short", index["arms"][name]["label"]) for name in arm_ids
    }
    return _training_rows(index, arm_ids, labels=labels)


def _variant_ids(index: dict[str, Any], reference: str = "convmixer") -> list[str]:
    def is_variant(name: str, arm: dict[str, Any]) -> bool:
        return (
            name == reference
            or index["paired_vs_reference"].get(name, {}).get("reference") == reference
        )

    return _arms(index, is_variant)


def _variant_labels(index: dict[str, Any], reference: str = "convmixer") -> dict[str, str]:
    """In a variant table the reference arm is the thing being varied against, not an arm."""
    labels = {name: index["arms"][name]["label"] for name in _variant_ids(index, reference)}
    labels[reference] = "published (reference)"
    return labels


def block_convmixer_variants(index: dict[str, Any]) -> str:
    arm_ids = _variant_ids(index)
    table = _training_rows(index, arm_ids, labels=_variant_labels(index))
    return table.replace("| model |", "| arm |", 1)


def block_convmixer_paired(index: dict[str, Any]) -> str:
    """Paired against the published arm, per the aggregation protocol in `docs/VALIDATION.md`.

    Percentile bootstrap resampling trajectories, seed 0. Stated because these intervals were
    previously computed as paired t-intervals (mean +/- t_7,0.975 * SE), which is a different
    estimator: it is symmetric by construction and on these eight pairs it is 10-20% wider.
    Every verdict is the same under both, but only one of them is the documented protocol.
    """
    labels = _variant_labels(index)
    rows = []
    for name in _variant_ids(index):
        paired = index["paired_vs_reference"].get(name)
        if paired is None or "mean_difference" not in paired:
            continue
        excludes = paired["excludes_zero"]
        worse = paired["mean_difference"] > 0
        if not excludes:
            verdict = "no"
        elif worse:
            # An interval excluding zero on the wrong side is still a resolved result, and
            # saying only "yes" would read as if the arm had won.
            verdict = "yes, *worse*"
        else:
            verdict = "**yes**"
        rows.append(
            [
                labels[name],
                _bold(f"{paired['mean_difference']:+.5f}", when=excludes and not worse),
                _ci(paired, "+.5f"),
                verdict,
                paired["arm_better_on"],
            ]
        )
    return _table(
        ["arm", "paired diff", "95% CI", "excludes 0", "arm better on"],
        ["---", "---:", "---", "---", "---"],
        rows,
    )


def _lead_time_table(
    index: dict[str, Any], arm_ids: list[str], labels: dict[str, str], *, first_column: str
) -> str:
    bands = index["lead_time_band_labels"]
    minima = {
        band: min(index["arms"][name]["test"]["lead_time_bands"][band] for name in arm_ids)
        for band in bands
    }
    rows = []
    for name in arm_ids:
        values = index["arms"][name]["test"]["lead_time_bands"]
        rows.append(
            [labels[name]]
            + [_bold(_f4(values[band]), when=values[band] == minima[band]) for band in bands]
        )
    return _table([first_column, *bands], ["---"] + ["---:"] * len(bands), rows)


def block_architectures_lead_time(index: dict[str, Any]) -> str:
    arm_ids = _arms(index, lambda _, arm: arm["role"] == "primary")
    labels = {
        name: index["arms"][name].get("short", index["arms"][name]["label"]) for name in arm_ids
    }
    return _lead_time_table(index, arm_ids, labels, first_column="model")


def block_convmixer_lead_time(index: dict[str, Any]) -> str:
    arm_ids = _variant_ids(index)
    labels = dict(_variant_labels(index))
    labels["convmixer"] = "published"
    return _lead_time_table(index, arm_ids, labels, first_column="arm")


# --------------------------------------------------------------------------------------------
# the documentation index
#
# These renderers read `docs/index.yaml` rather than the results index, and ignore their
# argument. The alternative -- threading a second context object through every renderer -- buys
# nothing: a block is a pure function of committed files either way.
# --------------------------------------------------------------------------------------------


def load_doc_registry(path: Path = DOC_REGISTRY) -> dict[str, Any]:
    registry: dict[str, Any] = yaml.safe_load(path.read_text())
    return registry


def _experiment_entries() -> list[dict[str, str]]:
    """Experiment write-ups, globbed and titled from their own heading.

    This is the category that grows with the work, so it is derived rather than listed: adding an
    experiment must not require editing an index, or the index is what will be forgotten.
    """
    entries = []
    for path in sorted((DOCS_ROOT / "experiments").glob("*.md")):
        named = EXPERIMENT_FILENAME.match(path.stem)
        if named is None:
            raise DriftError(
                f"docs/experiments/{path.name}: name it `<ID>-<slug>.md`, where ID is a TASKS.md "
                "task (`A-05`) or a decision (`D022`). The index takes the identifier from the "
                "filename, so an unnamed write-up cannot be listed against the work it belongs to."
            )
        identifier = named.group(1)
        heading = next(
            (line[2:].strip() for line in path.read_text().splitlines() if line.startswith("# ")),
            path.stem,
        )
        # The heading repeats the identifier the filename already carries; the table has a column
        # for it, so strip it rather than print it twice.
        summary = heading
        for separator in (" — ", " - "):
            if heading.startswith(identifier + separator):
                summary = heading[len(identifier + separator) :]
                break
        entries.append({"id": identifier, "file": f"experiments/{path.name}", "summary": summary})
    return entries


def documented_files() -> set[str]:
    """Every path the index accounts for, as `docs/`-relative strings."""
    listed = set()
    for section in load_doc_registry()["sections"]:
        for entry in section.get("files", []):
            listed.add(entry["file"])
    listed |= {entry["file"] for entry in _experiment_entries()}
    return listed


def block_doc_index(_: dict[str, Any]) -> str:
    """The whole index, grouped by kind, as rendered into `docs/README.md`."""
    parts: list[str] = []
    for section in load_doc_registry()["sections"]:
        entries = _experiment_entries() if section.get("glob") else list(section.get("files", []))
        if not entries:
            continue
        parts.append(f"### {section['title']}\n")
        note = " ".join(str(section.get("note", "")).split())
        if note:
            parts.append(f"{note}\n")
        has_ids = any("id" in entry for entry in entries)
        header = ["", "File", "Contents"] if has_ids else ["File", "Contents"]
        rows = [
            ([entry.get("id", "")] if has_ids else [])
            + [f"[`{entry['file']}`]({entry['file']})", entry["summary"]]
            for entry in entries
        ]
        parts.append(_table(header, ["---"] * len(header), rows) + "\n")
    return "\n".join(parts).rstrip("\n")


def block_reading_order(_: dict[str, Any]) -> str:
    """The contracts, in order, for a reader or agent about to change code."""
    lines = []
    position = 1
    for section in load_doc_registry()["sections"]:
        if not section.get("reading_order"):
            continue
        for entry in section["files"]:
            lines.append(f"{position}. `docs/{entry['file']}` — {entry['summary']}")
            position += 1
    lines.append(f"{position}. `TASKS.md` — task status and per-gate evidence")
    lines.append(
        f"{position + 1}. `docs/README.md` — the full index, including every experiment write-up"
    )
    return "\n".join(lines)


BLOCKS: dict[str, Callable[[dict[str, Any]], str]] = {
    "results:headline": block_headline,
    "results:frozen-test": block_frozen_test,
    "results:architectures": block_architectures,
    "results:architectures-lead-time": block_architectures_lead_time,
    "results:convmixer-variants": block_convmixer_variants,
    "results:convmixer-paired": block_convmixer_paired,
    "results:convmixer-lead-time": block_convmixer_lead_time,
    "docs:index": block_doc_index,
    "docs:reading-order": block_reading_order,
}


# --------------------------------------------------------------------------------------------
# block machinery
# --------------------------------------------------------------------------------------------


def markdown_files(root: Path = REPO_ROOT) -> list[Path]:
    """Tracked markdown, excluding the read-only solver submodule and agent definitions."""
    paths = [root / "README.md", root / "CLAUDE.md", root / "TASKS.md"]
    paths += sorted(DOCS_ROOT.rglob("*.md"))
    return [path for path in paths if path.is_file()]


def _apply(path: Path, index: dict[str, Any], *, rewrite: bool) -> tuple[str, list[str]]:
    """Return the file's rendered text and a report line per block that drifted."""
    original = path.read_text()
    lines = original.splitlines()
    output: list[str] = []
    problems: list[str] = []
    cursor = 0
    fenced = False
    while cursor < len(lines):
        line = lines[cursor]
        if line.startswith("```"):
            fenced = not fenced
        # A marker inside a fenced block is documentation *about* markers -- `docs/DOCUMENTATION.md`
        # shows both kinds as examples. Rendering into it would fill an illustration with a real
        # table, and reporting it would make the file that explains the mechanism fail the check.
        opening = None if fenced else BEGIN.match(line)
        if opening is None:
            output.append(line)
            cursor += 1
            continue

        kind, block_id = opening.group(1), opening.group(2)
        closing = None
        for offset in range(cursor + 1, len(lines)):
            candidate = END.match(lines[offset])
            if candidate is not None:
                closing = offset
                break
        if closing is None:
            raise DriftError(f"{path.name}: block {block_id!r} is opened but never closed")
        if (END.match(lines[closing]).group(1), END.match(lines[closing]).group(2)) != (  # type: ignore[union-attr]
            kind,
            block_id,
        ):
            raise DriftError(
                f"{path.name}: block {block_id!r} ({kind}) is closed by "
                f"{lines[closing]!r}; markers must match"
            )
        if block_id not in BLOCKS:
            raise DriftError(
                f"{path.name}: no renderer for block {block_id!r}. Add one to BLOCKS in "
                "swe_sr/docgen.py, or remove the markers."
            )

        current = "\n".join(lines[cursor + 1 : closing]).strip("\n")
        rendered = BLOCKS[block_id](index).strip("\n")
        if current != rendered:
            diff = "\n".join(
                difflib.unified_diff(
                    current.splitlines(),
                    rendered.splitlines(),
                    fromfile=f"{path.name}:{block_id} (committed)",
                    tofile=f"{path.name}:{block_id} (rendered from index)",
                    lineterm="",
                )
            )
            if kind == "verified":
                problems.append(
                    f"{_display(path)}: verified block {block_id!r} no longer "
                    f"matches the artifacts. This block is a freeze record and is never "
                    f"rewritten: if the change is real it needs a new decision and a new "
                    f"freeze, not an edit.\n{diff}"
                )
            elif not rewrite:
                problems.append(
                    f"{_display(path)}: generated block {block_id!r} is stale; run "
                    f"`python -m swe_sr.docgen render`.\n{diff}"
                )

        output.append(line)
        keep = kind == "verified" or not rewrite
        output.extend(lines[cursor + 1 : closing] if keep else rendered.splitlines())
        output.append(lines[closing])
        cursor = closing + 1

    text = "\n".join(output)
    if original.endswith("\n"):
        text += "\n"
    return text, problems


def render(index: dict[str, Any], *, rewrite: bool) -> tuple[list[Path], list[str]]:
    changed: list[Path] = []
    problems: list[str] = []
    for path in markdown_files():
        text, issues = _apply(path, index, rewrite=rewrite)
        problems.extend(issues)
        if rewrite and text != path.read_text():
            path.write_text(text)
            changed.append(path)
    return changed, problems


def block_ids_in_use() -> dict[str, list[str]]:
    """Block id -> the documents containing it. Fenced examples do not count as uses."""
    found: dict[str, list[str]] = {}
    for path in markdown_files():
        fenced = False
        for line in path.read_text().splitlines():
            if line.startswith("```"):
                fenced = not fenced
                continue
            opening = None if fenced else BEGIN.match(line)
            if opening is not None:
                found.setdefault(opening.group(2), []).append(_display(path))
    return found


# --------------------------------------------------------------------------------------------
# hygiene lint
# --------------------------------------------------------------------------------------------


def lint(paths: list[Path] | None = None) -> list[str]:
    """Checks that catch the drift a generated table cannot: stale paths, orphans, bad refs.

    `paths` exists so the rules can be tested against fixtures rather than only against this
    repository, where they pass by construction and a broken rule would look like a working one.
    """
    problems: list[str] = []
    texts = {path: path.read_text() for path in (paths if paths is not None else markdown_files())}

    # 1. Backticked repository paths must resolve. A renamed or moved document otherwise leaves
    #    live-looking pointers behind in every file that mentioned it.
    #
    #    Only paths rooted in a directory this repository actually has are checked. That is not
    #    laziness: `docs/RESEARCH_MATRIX.md` records findings against *other* repositories, and
    #    `space_time_pde/src/unet.py` is a correct citation of an upstream file that by
    #    construction does not exist here. Paths containing an elision (`.../manifest.json`) are
    #    skipped for the same reason -- they name a shape, not a file.
    top_level = {entry.name for entry in REPO_ROOT.iterdir() if entry.is_dir()}
    for path, text in texts.items():
        for match in sorted(set(BACKTICKED_PATH.findall(text))):
            if "..." in match or match.split("/", 1)[0] not in top_level:
                continue
            if not (REPO_ROOT / match).exists():
                problems.append(f"{_display(path)}: references `{match}`, which does not exist")

    # 2. The index accounts for every document, in both directions. This is the rule that fixes
    #    what started all of it: `README.md`'s hand-maintained table had silently omitted two
    #    write-ups that its own prose linked to.
    indexed = documented_files()
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        relative = str(path.relative_to(DOCS_ROOT))
        if relative == "README.md":
            continue  # the index itself
        if relative not in indexed:
            problems.append(
                f"docs/{relative}: not in docs/index.yaml. Add it to a section, or -- if it is an "
                "experiment write-up -- move it to docs/experiments/, where it is picked up "
                "automatically."
            )
    for entry in sorted(indexed):
        if not (DOCS_ROOT / entry).exists():
            problems.append(f"docs/index.yaml lists `{entry}`, which does not exist")

    # 3. Every decision cited must exist, so a reference like D021 cannot outlive a renumbering.
    if DECISION_LOG.is_file():
        defined = set(DECISION_HEADING.findall(DECISION_LOG.read_text()))
        for path, text in texts.items():
            if path == DECISION_LOG:
                continue
            for reference in sorted(set(DECISION_REF.findall(text))):
                if reference not in defined:
                    problems.append(
                        f"{_display(path)}: cites {reference}, which has no entry in "
                        f"docs/DECISIONS.md"
                    )

    # 4. Every run ID cited must be registered, so a number can always be traced to an arm the
    #    index knows how to rebuild.
    registered = {arm["run_id"] for arm in load_registry()["arms"]}
    for path, text in texts.items():
        for run_id in sorted(set(RUN_ID.findall(text))):
            if run_id not in registered:
                problems.append(
                    f"{_display(path)}: cites run `{run_id}`, which is not in "
                    "docs/results/runs.yaml. Register it, so its numbers are rendered rather "
                    "than transcribed."
                )

    return problems


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist; run `python -m swe_sr.results --write` (needs runs/)"
        )
    index: dict[str, Any] = json.loads(path.read_text())
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "check", "lint"))
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args(argv)

    problems: list[str] = []
    if args.command in ("render", "check"):
        index = load_index(args.index)
        changed, problems = render(index, rewrite=args.command == "render")
        if args.command == "render":
            for path in changed:
                print(f"updated {_display(path)}")
            if not changed:
                print("all generated blocks already current")
        else:
            print(f"checked {sum(len(v) for v in block_ids_in_use().values())} blocks")

    if args.command in ("check", "lint"):
        problems.extend(lint())
        unused = sorted(set(BLOCKS) - set(block_ids_in_use()))
        if unused:
            problems.append(
                f"renderers with no block in any document: {unused}. Either place the markers or "
                "remove the renderer; an unused renderer is untested."
            )

    if problems:
        print(f"\n{len(problems)} problem(s):\n", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}\n", file=sys.stderr)
        return 1
    if args.command != "render":
        print("documentation is consistent with docs/results/index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
