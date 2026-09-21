# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Public-contract tests for the GBOpt.GBMinimizer compatibility facade (#62)."""

from GBOpt import GBMinimizer
from GBOpt.optimization.genetic import ENERGY_PENALTY, GeneticAlgorithmMinimizer
from GBOpt.optimization.monte_carlo import MonteCarloMinimizer
from GBOpt.optimization.mutation import Mutator
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
)


def test_gbminimizer_reexports_resolve_to_the_canonical_optimization_objects():
    assert GBMinimizer.GBMinimizerError is GBMinimizerError
    assert GBMinimizer.GBMinimizerTypeError is GBMinimizerTypeError
    assert GBMinimizer.GBMinimizerValueError is GBMinimizerValueError
    assert GBMinimizer.MonteCarloMinimizer is MonteCarloMinimizer
    assert GBMinimizer.GeneticAlgorithmMinimizer is GeneticAlgorithmMinimizer
    assert GBMinimizer.Mutator is Mutator
    assert GBMinimizer.ENERGY_PENALTY is ENERGY_PENALTY
