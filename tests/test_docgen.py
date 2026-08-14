"""Tests for the documentation renderer and hygiene lint (D024).

The point of `swe_sr.docgen` is that a stale number fails rather than rots, so the tests that
matter most are the negative ones: a wrong table must be *detected*, and a freeze block must be
detected without being repaired. A test that only asserts the current documents pass would go
green against a renderer that reported nothing at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_sr import docgen
from swe_sr.docgen import DriftError

pytestmark = pytest.mark.filterwarnings("error")

INDEX_PRESENT = docgen.INDEX_PATH.is_file()
needs_index = pytest.mark.skipif(not INDEX_PRESENT, reason="docs/results/index.json not built")


@pytest.fixture
def index() -> dict[str, object]:
    return docgen.load_index()


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(body)
    return path


# ------------------------------------------------------------------------------------------
# the documents in this repository
# ------------------------------------------------------------------------------------------


@needs_index
def test_committed_documents_match_the_index(index: dict[str, object]) -> None:
    """Every generated and verified block in the repository is current."""
    _, problems = docgen.render(index, rewrite=False)
    assert problems == []


def test_lint_passes_on_this_repository() -> None:
    assert docgen.lint() == []


@needs_index
def test_every_renderer_is_actually_used(index: dict[str, object]) -> None:
    """An unplaced renderer is untested, and its output is nowhere a reader can see it."""
    assert set(docgen.BLOCKS) == set(docgen.block_ids_in_use())


@needs_index
def test_headline_table_reports_the_frozen_result(index: dict[str, object]) -> None:
    """Pin one number end to end, so a renderer that silently emitted nothing would fail."""
    table = docgen.block_headline(index)
    assert "| U-Net x4 | 1,930,208 | **0.0400** | [0.0261, 0.0544] |" in table
    assert "| bicubic | 0 | 0.4295 |" in table
    # Worst to best, so the winner ends the table.
    assert table.index("EDSR x4") < table.index("ConvMixer x4") < table.index("U-Net x4")


@needs_index
def test_only_the_column_minimum_is_bolded(index: dict[str, object]) -> None:
    """Bolding is a rule over the whole column, baselines included, not an editorial choice."""
    table = docgen.block_frozen_test(index)
    # bicubic beats EDSR on mass error, so EDSR is not bolded for winning among models only.
    assert "**0.0540**" not in table
    assert "**0.0347**" in table


@needs_index
def test_lead_time_bands_keep_the_registry_order(index: dict[str, object]) -> None:
    """The index is serialized sorted, which would otherwise put "16-24 h" first."""
    header = docgen.block_architectures_lead_time(index).splitlines()[0]
    assert header == "| model | <= 12 h | 16-24 h | > 24 h |"


# ------------------------------------------------------------------------------------------
# drift detection
# ------------------------------------------------------------------------------------------


@needs_index
def test_stale_generated_block_is_reported_and_then_repaired(
    tmp_path: Path, index: dict[str, object]
) -> None:
    path = _write(
        tmp_path,
        "intro\n"
        "<!-- BEGIN generated: results:headline -->\n"
        "| Method | Params | normMSE | 95% CI |\n"
        "| U-Net x4 | 1,930,208 | 0.9999 | [0.0, 1.0] |\n"
        "<!-- END generated: results:headline -->\n"
        "outro\n",
    )

    _, problems = docgen._apply(path, index, rewrite=False)
    assert len(problems) == 1
    assert "is stale" in problems[0]
    assert "0.9999" in problems[0]

    text, _ = docgen._apply(path, index, rewrite=True)
    path.write_text(text)
    assert "0.9999" not in text
    assert "**0.0400**" in text
    assert text.startswith("intro\n") and text.endswith("outro\n")

    _, problems = docgen._apply(path, index, rewrite=False)
    assert problems == []


@needs_index
def test_verified_block_is_reported_but_never_rewritten(
    tmp_path: Path, index: dict[str, object]
) -> None:
    """A freeze record is not repaired to agree with the artifacts; that is its whole purpose."""
    body = (
        "<!-- BEGIN verified: results:frozen-test -->\n"
        "| Method | Params | normMSE |\n"
        "| U-Net | 1,930,208 | 0.9999 |\n"
        "<!-- END verified: results:frozen-test -->\n"
    )
    path = _write(tmp_path, body)

    for rewrite in (False, True):
        text, problems = docgen._apply(path, index, rewrite=rewrite)
        assert len(problems) == 1
        assert "never rewritten" in problems[0]
        assert text == body, "a verified block must survive `render` untouched"


@needs_index
def test_unclosed_mismatched_and_unknown_blocks_all_fail(
    tmp_path: Path, index: dict[str, object]
) -> None:
    cases = {
        "opened but never closed": "<!-- BEGIN generated: results:headline -->\nbody\n",
        "markers must match": (
            "<!-- BEGIN generated: results:headline -->\n<!-- END verified: results:headline -->\n"
        ),
        "no renderer for block": (
            "<!-- BEGIN generated: results:invented -->\n<!-- END generated: results:invented -->\n"
        ),
    }
    for expected, body in cases.items():
        with pytest.raises(DriftError, match=expected):
            docgen._apply(_write(tmp_path, body), index, rewrite=False)


# ------------------------------------------------------------------------------------------
# hygiene rules, against fixtures rather than only against a repository that passes
# ------------------------------------------------------------------------------------------


def test_lint_flags_a_path_that_does_not_exist(tmp_path: Path) -> None:
    path = _write(tmp_path, "see `docs/NOT_A_REAL_DOC.md` for detail\n")
    assert any("NOT_A_REAL_DOC" in problem for problem in docgen.lint([path]))


def test_lint_ignores_paths_outside_this_repository(tmp_path: Path) -> None:
    """`docs/RESEARCH_MATRIX.md` cites upstream files that correctly do not exist here."""
    path = _write(
        tmp_path,
        "upstream `space_time_pde/src/unet.py`, elided `data/staging/.../manifest.json`, "
        "placeholder `runs/<run-id>/summary.json`\n",
    )
    assert docgen.lint([path]) == []


def test_lint_flags_an_undefined_decision(tmp_path: Path) -> None:
    path = _write(tmp_path, "as required by D099\n")
    problems = docgen.lint([path])
    assert any("D099" in problem for problem in problems)


def test_lint_flags_an_unregistered_run(tmp_path: Path) -> None:
    path = _write(tmp_path, "measured on `20260101T000000Z_unet_deadbeef_cafebabe`\n")
    problems = docgen.lint([path])
    assert any("runs.yaml" in problem for problem in problems)


def test_lint_accepts_a_registered_run(tmp_path: Path) -> None:
    registered = next(iter(docgen.load_registry()["arms"]))["run_id"]
    assert docgen.lint([_write(tmp_path, f"`{registered}`\n")]) == []


# ------------------------------------------------------------------------------------------
# the documentation index
# ------------------------------------------------------------------------------------------


def test_experiment_write_ups_are_globbed_not_listed() -> None:
    """Adding an experiment must not require editing an index, or the index is what gets missed."""
    listed = {
        entry["file"]
        for section in docgen.load_doc_registry()["sections"]
        for entry in section.get("files", [])
    }
    globbed = {entry["file"] for entry in docgen._experiment_entries()}
    assert globbed, "no experiment write-ups found"
    assert not (globbed & listed), "experiment write-ups must not also be listed by hand"


def test_every_document_is_accounted_for_in_both_directions() -> None:
    """The rule that fixes the original bug: README's table had omitted two live write-ups."""
    indexed = docgen.documented_files()
    on_disk = {
        str(path.relative_to(docgen.DOCS_ROOT))
        for path in docgen.DOCS_ROOT.rglob("*.md")
        if path.name != "README.md"
    }
    assert on_disk <= indexed
    assert all((docgen.DOCS_ROOT / entry).exists() for entry in indexed)


def test_a_write_up_without_an_identifier_is_rejected(tmp_path: Path) -> None:
    stray = docgen.DOCS_ROOT / "experiments" / "notes-on-something.md"
    stray.write_text("# Notes\n")
    try:
        with pytest.raises(DriftError, match="name it"):
            docgen._experiment_entries()
    finally:
        stray.unlink()


def test_markers_inside_a_code_fence_are_examples_not_blocks(
    tmp_path: Path, index: dict[str, object]
) -> None:
    """`docs/DOCUMENTATION.md` shows both marker kinds as illustrations; filling them is wrong."""
    body = (
        "```markdown\n"
        "<!-- BEGIN generated: results:headline -->\n"
        "<!-- END generated: results:headline -->\n"
        "```\n"
    )
    path = _write(tmp_path, body)
    text, problems = docgen._apply(path, index, rewrite=True)
    assert text == body
    assert problems == []


# ------------------------------------------------------------------------------------------
# the index the renderers read
# ------------------------------------------------------------------------------------------


@needs_index
def test_committed_index_is_canonically_formatted() -> None:
    """Serialization must be stable, or every rebuild produces a diff that means nothing."""
    from swe_sr.results import render_index

    committed = docgen.INDEX_PATH.read_text()
    assert render_index(json.loads(committed)) == committed


@needs_index
def test_index_excludes_host_dependent_timings(index: dict[str, object]) -> None:
    """`seconds_per_frame` is wall clock under unknown load; it has already misled once."""
    assert "seconds_per_frame" not in json.dumps(index)
