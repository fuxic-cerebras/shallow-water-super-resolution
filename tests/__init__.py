"""Test package.

This file is deliberately present. Under pytest's default `prepend` import mode, the
directory added to `sys.path` is the first ancestor *without* an `__init__.py`. Without
this file that ancestor is `tests/` itself, so `tests.solver.reference_harness` is not
importable and the parity tests fail to collect under a bare `pytest` invocation -- even
though they pass under `python -m pytest`, which happens to put the repository root on
`sys.path` anyway. Making `tests` a real package means both invocations behave the same,
and local runs match CI.
"""
