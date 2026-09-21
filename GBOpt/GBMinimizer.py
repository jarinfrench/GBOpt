# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Compatibility facade over :mod:`GBOpt.optimization`.

The canonical implementations of :class:`Mutator`, :class:`MonteCarloMinimizer`, and
:class:`GeneticAlgorithmMinimizer` live under :mod:`GBOpt.optimization`. This module
re-exports the same class and exception objects so that existing imports of
``GBOpt.GBMinimizer`` continue to resolve unchanged.

New code should import directly from :mod:`GBOpt.optimization` (for example,
``from GBOpt.optimization import MonteCarloMinimizer``); this module is retained only
for import-path compatibility and may be removed in a future, separately approved
change.
"""

from GBOpt.optimization import (
    ENERGY_PENALTY,
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
    GeneticAlgorithmMinimizer,
    MonteCarloMinimizer,
    Mutator,
)

__all__ = [
    "ENERGY_PENALTY",
    "GBMinimizerError",
    "GBMinimizerTypeError",
    "GBMinimizerValueError",
    "GeneticAlgorithmMinimizer",
    "MonteCarloMinimizer",
    "Mutator",
]
