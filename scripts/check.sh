#!/usr/bin/env bash
# Run exactly what CI runs, in the same order, and fail on the first error.
#
# This exists because a local check that only *looks* clean is worse than no check. Two
# ways that happened here:
#
#   1. `.gitignore` contained an unanchored `data/`, which matches any directory named
#      data at any depth. Ruff respects .gitignore, so `swe_sr/data/` and `tests/data/`
#      were skipped entirely and lint reported success over 17 of 25 files.
#   2. Piping each tool through `tail` discards its exit status: the pipeline reports
#      tail's success, so a failing check can read as passing.
#
# So: no pipes, no tail, `set -e`, and the same command strings as .github/workflows/ci.yml.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f references/shallow-water/swe.py ]]; then
    echo "FAIL: solver submodule missing; parity tests cannot run." >&2
    echo "      run: git submodule update --init" >&2
    exit 1
fi

# Guard the root cause directly: if a bare `data/`-style pattern ever comes back, the
# data layer silently drops out of every lint run again.
if git check-ignore -q swe_sr/data 2>/dev/null; then
    echo "FAIL: .gitignore excludes swe_sr/data. Anchor generated-output patterns with a" >&2
    echo "      leading slash (/data/*), or ruff will skip the data layer." >&2
    exit 1
fi

# Independently confirm ruff's file count, since the failure mode is silent skipping
# rather than an error.
tracked=$(git ls-files '*.py' | grep -c -E '^(swe_sr|tests)/' || true)
echo "== tracked python files under swe_sr/ and tests/: ${tracked}"

echo "== ruff check"
ruff check swe_sr tests scripts

echo "== ruff format --check"
ruff format --check swe_sr tests scripts

echo "== mypy"
mypy

echo "== pytest (fast suite)"
pytest -q -m "not slow and not dataset"

echo
echo "ALL CHECKS PASSED"
