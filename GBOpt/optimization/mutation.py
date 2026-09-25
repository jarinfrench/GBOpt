# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Legacy-compatibility ``choices: list[str]`` adapter over OperationSpec dispatch.

``Mutator`` keeps its established public shape (``choices_keys``,
``mutate(local_random, GB, manipulator)``) so existing ``MonteCarloMinimizer``/
``GeneticAlgorithmMinimizer`` callers configured with the pre-OperationSpec
``choices: list[str]`` argument keep working, with the same defaults and the same
fixed-seed histories. Internally, selection is now OperationSpec-driven
(``GBOpt.optimization.dispatch``) instead of a hard-coded name ``match``/``case``; only
the per-operation invocation (calling the corresponding established
``GBManipulator`` method directly, bypassing ``ManipulationContext`` -- see
``GBOpt.optimization.dispatch``'s module docstring for why) still needs a small
name-keyed table, since that mapping is unavoidable however dispatch is structured.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt.manipulation import AtomInsertion, AtomRemoval, RightGrainTranslation, Manipulation
from GBOpt.optimization.dispatch import LegacyInvoker, run_legacy_compat_operation
from GBOpt.optimization.types import GBMinimizerError, GBMinimizerValueError, OperationSpec

# The only three operation names the pre-OperationSpec `choices: list[str]` vocabulary
# ever supported (Mutator._apply_mutation's former match/case), each mapped to the
# built-in operation whose arity backs its OperationSpec bookkeeping.
_LEGACY_OPERATION_TYPES: dict[str, type[Manipulation]] = {
    "insert_atoms": AtomInsertion,
    "remove_atoms": AtomRemoval,
    "translate_right_grain": RightGrainTranslation,
}


def _insert_atoms_invoker(manipulator: GBManipulator, rng: np.random.Generator):
    del rng
    new_system = manipulator.insert_atoms(method="grid", num_to_insert=1)
    return "add1", new_system


def _remove_atoms_invoker(manipulator: GBManipulator, rng: np.random.Generator):
    del rng
    new_system = manipulator.remove_atoms(num_to_remove=1)
    return "remove1", new_system


def _make_translate_right_grain_invoker(GB: GBMaker) -> LegacyInvoker:
    def _invoke(manipulator: GBManipulator, rng: np.random.Generator):
        parent = manipulator.parents[0]
        y_dim = parent.box_dims[1, 1] - parent.box_dims[1, 0]
        z_dim = parent.box_dims[2, 1] - parent.box_dims[2, 0]

        dy = (y_dim / GB.repeat_factor[0]) * rng.uniform(0, 1)
        dz = (z_dim / GB.repeat_factor[1]) * rng.uniform(0, 1)

        new_system = manipulator.translate_right_grain(dy=dy, dz=dz)
        return f"shift{dy:.8f}dy{dz:.8f}dz", new_system

    return _invoke


_LEGACY_INVOKER_FACTORIES: dict[str, Callable[[GBMaker], LegacyInvoker]] = {
    "insert_atoms": lambda GB: _insert_atoms_invoker,
    "remove_atoms": lambda GB: _remove_atoms_invoker,
    "translate_right_grain": _make_translate_right_grain_invoker,
}


class Mutator:
    """Perform randomly selected manipulations on a GB candidate.

    :param choices: Mutation operation names to make available.
    :param manipulator: GBManipulator used to validate the requested operations.
    """

    # TODO: Add more manipulator options to this class as we make more
    # manipulators faster.

    def __init__(self, choices: list[str], manipulator: GBManipulator):
        invalid_choices = [
            method for method in choices if not hasattr(manipulator, method)
        ]
        if invalid_choices:
            raise GBMinimizerValueError(
                "Unknown GBManipulator mutation choice(s): "
                + ", ".join(repr(choice) for choice in invalid_choices)
            )

        # Duplicate names do not weight a mutation more heavily.
        self.choices_keys = list(dict.fromkeys(choices))
        if not self.choices_keys:
            raise GBMinimizerValueError(
                "At least one mutation choice must be provided."
            )

    def mutate(
        self,
        local_random: np.random.Generator,
        GB: GBMaker,
        manipulator: GBManipulator,
    ):
        """Perform a randomly selected feasible mutation.

        Each configured mutation is attempted at most once, in the same
        equal-weight/permutation order the pre-OperationSpec implementation used. If an
        operation is physically infeasible for the current candidate, another configured
        operation is tried. Failure of every configured operation is fatal.

        :param local_random: Optimizer-owned random-number generator.
        :param GB: GBMaker providing boundary dimensions and repeat factors.
        :param manipulator: GBManipulator on which to perform the mutation.
        :return: Mutation description and resulting atom positions.
        :raises GBMinimizerValueError: If a configured choice is not one of this
            adapter's known operation names.
        :raises GBMinimizerError: If no configured mutation can produce a candidate.
        """
        specs: list[OperationSpec] = []
        legacy_invokers: dict[str, LegacyInvoker] = {}
        for name in self.choices_keys:
            if name not in _LEGACY_OPERATION_TYPES:
                raise GBMinimizerValueError(f"Unhandled mutation choice: {name!r}")
            specs.append(
                OperationSpec(
                    name=name,
                    operation=_LEGACY_OPERATION_TYPES[name](),
                    weight=1.0,
                )
            )
            legacy_invokers[name] = _LEGACY_INVOKER_FACTORIES[name](GB)

        return run_legacy_compat_operation(
            specs,
            rng=local_random,
            manipulator=manipulator,
            legacy_invokers=legacy_invokers,
        )
