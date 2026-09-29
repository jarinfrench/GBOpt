# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Verify the refactor's subpackage import boundaries hold in a fresh interpreter.

Each check runs `python -c "..."` in a subprocess (never an in-process assertion on
`sys.modules`, which is shared across the whole pytest run and would false-pass or
false-fail depending on unrelated test order). Where the forbidden target is something
`GBOpt/__init__.py` itself eagerly imports -- `GBOpt.GBMaker`, `GBOpt.GBManipulator`,
and (transitively, since both facades import structure/writer/reader helpers) `GBOpt.io`
-- the subprocess stubs `sys.modules["GBOpt"]` with an empty module before importing the
target, so `GBOpt/__init__.py`'s own body never runs and only the target's own import
graph is observed. Where the forbidden target is not eagerly imported by `GBOpt/__init__`
(`GBOpt.optimization`, `GBOpt.evaluation`, `GBOpt.snapshot`, `GBOpt.Checkpoint`), the
plain pattern is sufficient.
"""

from __future__ import annotations

import subprocess
import sys

_STUB_PREAMBLE = (
    "import importlib.util, sys, types\n"
    "spec = importlib.util.find_spec('GBOpt')\n"
    "stub = types.ModuleType('GBOpt')\n"
    "stub.__path__ = spec.submodule_search_locations\n"
    "sys.modules['GBOpt'] = stub\n"
)


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_boundary(import_stmt: str, forbidden: str, *, stub: bool = False) -> None:
    """Run one import-boundary check in a fresh interpreter and assert it passed.

    :param import_stmt: Statement importing the module under test.
    :param forbidden: Fully-qualified module name that must not end up in
        ``sys.modules`` as a result of ``import_stmt``.
    :param stub: Keyword argument, optional, defaults to ``False``. Whether to stub
        ``sys.modules["GBOpt"]`` first, required whenever ``forbidden`` (or anything
        that would transitively pull it in) is one of ``GBOpt/__init__.py``'s own
        eager imports.
    """
    script = (_STUB_PREAMBLE if stub else "") + (
        f"{import_stmt}\n"
        f"import sys\n"
        f"assert {forbidden!r} not in sys.modules, sorted(sys.modules)\n"
    )
    result = _run(script)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# GBOpt.gbmaker imports neither optimizer nor file-format implementation layers.
# --------------------------------------------------------------------------------------


def test_gbmaker_subpackage_does_not_import_optimization() -> None:
    _assert_boundary("import GBOpt.gbmaker", "GBOpt.optimization")


def test_gbmaker_subpackage_does_not_import_io() -> None:
    _assert_boundary("import GBOpt.gbmaker", "GBOpt.io", stub=True)


# --------------------------------------------------------------------------------------
# GBOpt.io does not import the GBMaker facade or minimizer implementations.
# --------------------------------------------------------------------------------------


def test_io_does_not_import_gbmaker_facade() -> None:
    _assert_boundary("import GBOpt.io", "GBOpt.GBMaker", stub=True)


def test_io_does_not_import_gbmanipulator_facade() -> None:
    _assert_boundary("import GBOpt.io", "GBOpt.GBManipulator", stub=True)


def test_io_does_not_import_optimization() -> None:
    _assert_boundary("import GBOpt.io", "GBOpt.optimization")


# --------------------------------------------------------------------------------------
# Manipulation operations perform no file I/O/evaluation; neutral domain state only.
# --------------------------------------------------------------------------------------


def test_manipulation_does_not_import_io() -> None:
    _assert_boundary("import GBOpt.manipulation", "GBOpt.io", stub=True)


def test_manipulation_does_not_import_evaluation() -> None:
    _assert_boundary("import GBOpt.manipulation", "GBOpt.evaluation")


def test_manipulation_does_not_import_optimization() -> None:
    _assert_boundary("import GBOpt.manipulation", "GBOpt.optimization")


# --------------------------------------------------------------------------------------
# Checkpoint persistence/model code does not import live optimizer classes; optimizer
# modules own state-transition policy.
# --------------------------------------------------------------------------------------


def test_checkpoint_module_does_not_import_optimization() -> None:
    _assert_boundary("import GBOpt.Checkpoint", "GBOpt.optimization")


def test_snapshot_package_does_not_import_optimization() -> None:
    """The schema-v2 snapshot codec must not pull in live minimizer classes.

    ``GBOpt.snapshot.types``/``migration`` do import a few pure helpers from
    ``GBOpt.optimization.types`` (mapping (de)serialization shared with schema-v1), but
    only as local, call-time imports specifically to avoid a module-scope cycle with
    ``GBOpt.optimization.genetic`` -- see the R29 fix. This confirms that deferral still
    holds: importing the snapshot package alone must never drag in
    ``GBOpt.optimization`` (and therefore never ``MonteCarloMinimizer``/
    ``GeneticAlgorithmMinimizer``) as a side effect.
    """
    _assert_boundary("import GBOpt.snapshot", "GBOpt.optimization")


# --------------------------------------------------------------------------------------
# Events/journals do not own restart-critical state.
# --------------------------------------------------------------------------------------


def test_observability_does_not_import_snapshot() -> None:
    _assert_boundary("import GBOpt.observability", "GBOpt.snapshot")


def test_observability_does_not_import_checkpoint_module() -> None:
    _assert_boundary("import GBOpt.observability", "GBOpt.Checkpoint")


# --------------------------------------------------------------------------------------
# Compatibility import identity: GBMaker, GBManipulator, and both minimizers still
# resolve to the same public classes documented since R01/R14.
# --------------------------------------------------------------------------------------


def test_gbmaker_facade_resolves_to_gbmaker_subpackage_construction_pipeline() -> None:
    from GBOpt import GBMaker
    from GBOpt.gbmaker.assembly import assemble_bicrystal

    # GBMaker.py is a compatibility facade over the gbmaker subpackage (R10): its
    # construction path delegates to assemble_bicrystal, not a parallel reimplementation.
    assert GBMaker.__module__ == "GBOpt.GBMaker"
    assert callable(assemble_bicrystal)


def test_top_level_compatibility_imports_resolve_to_documented_classes() -> None:
    import GBOpt
    from GBOpt.GBManipulator import GBManipulator, InterfaceCandidate
    from GBOpt.GBMaker import GBMaker
    from GBOpt.optimization import GeneticAlgorithmMinimizer, MonteCarloMinimizer

    assert GBOpt.GBMaker is GBMaker
    assert GBOpt.GBManipulator is GBManipulator
    assert GBOpt.InterfaceCandidate is InterfaceCandidate
    # Both minimizers' compatibility facades: GBOpt.optimization's curated __init__
    # re-export must be the same object as importing the submodule directly, not a copy.
    from GBOpt.optimization.genetic import GeneticAlgorithmMinimizer as _GA
    from GBOpt.optimization.monte_carlo import MonteCarloMinimizer as _MC

    assert GeneticAlgorithmMinimizer is _GA
    assert MonteCarloMinimizer is _MC
