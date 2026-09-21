# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Import-compatibility tests for the GBOpt.optimization decomposition (R01/#62).

These tests verify that splitting GBOpt/GBMinimizer.py into GBOpt/optimization/
submodules did not change the identity of any publicly re-exported class or
exception object. GBOpt.GBMinimizer must remain a compatibility facade.
"""

from GBOpt import GBMinimizer
from GBOpt.optimization import checkpointing, errors, evaluation, genetic
from GBOpt.optimization import monte_carlo as monte_carlo_module
from GBOpt.optimization import mutation as mutation_module


def test_gbminimizer_reexports_are_identical_objects():
    from GBOpt.optimization.errors import (
        GBMinimizerError,
        GBMinimizerTypeError,
        GBMinimizerValueError,
    )
    from GBOpt.optimization.genetic import ENERGY_PENALTY, GeneticAlgorithmMinimizer
    from GBOpt.optimization.monte_carlo import MonteCarloMinimizer
    from GBOpt.optimization.mutation import Mutator

    assert GBMinimizer.GBMinimizerError is GBMinimizerError
    assert GBMinimizer.GBMinimizerTypeError is GBMinimizerTypeError
    assert GBMinimizer.GBMinimizerValueError is GBMinimizerValueError
    assert GBMinimizer.MonteCarloMinimizer is MonteCarloMinimizer
    assert GBMinimizer.GeneticAlgorithmMinimizer is GeneticAlgorithmMinimizer
    assert GBMinimizer.Mutator is Mutator
    assert GBMinimizer.ENERGY_PENALTY is ENERGY_PENALTY


def test_gbminimizer_facade_contains_no_canonical_implementations():
    """GBOpt/GBMinimizer.py must hold only imports/re-exports, per issue #62."""
    import ast
    import inspect

    source = inspect.getsource(GBMinimizer)
    tree = ast.parse(source)
    disallowed = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert disallowed == []


def test_exception_hierarchy_unchanged():
    assert issubclass(errors.GBMinimizerTypeError, errors.GBMinimizerError)
    assert issubclass(errors.GBMinimizerTypeError, TypeError)
    assert issubclass(errors.GBMinimizerValueError, errors.GBMinimizerError)
    assert issubclass(errors.GBMinimizerValueError, ValueError)


def test_old_style_module_attribute_lookup_resolves_like_pickle_find_class():
    """Simulate how pickle.Unpickler.find_class resolves a legacy pickled path.

    Historical checkpoints may reference callables as
    ``GBOpt.GBMinimizer.MonteCarloMinimizer`` / ``GeneticAlgorithmMinimizer``. Pickle
    resolves such references with ``getattr(sys.modules[module], name)``, which this
    test reproduces directly against the live facade module.
    """
    import sys

    module = sys.modules["GBOpt.GBMinimizer"]
    assert getattr(module, "MonteCarloMinimizer") is monte_carlo_module.MonteCarloMinimizer
    assert (
        getattr(module, "GeneticAlgorithmMinimizer")
        is genetic.GeneticAlgorithmMinimizer
    )
    assert getattr(module, "Mutator") is mutation_module.Mutator


def test_optimization_submodules_are_importable():
    for module in (checkpointing, errors, evaluation, genetic, monte_carlo_module,
                   mutation_module):
        assert module is not None
