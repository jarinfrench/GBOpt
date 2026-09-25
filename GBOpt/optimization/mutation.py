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

A ``choices`` entry that isn't one of the three legacy names is resolved by registry
lookup (``GBOpt.manipulation.default_registry`` unless a caller supplies another) and
dispatched through the generic ``GBManipulator.apply()``/``ManipulationContext``
boundary instead -- this is how a third-party operation participates in MC/GA without
editing this module: ``registry.register("my_op", MyOp())``, then add ``"my_op"`` to
``choices``. Such an operation has no legacy tolerance to preserve, so the stricter,
uniform boundary is safe for it. It must produce exactly one child; a different count
is a fatal ``GBMinimizerValueError`` (not retried as mere infeasibility), since it
indicates the operation itself is incompatible with MC/GA's single-child contract,
not that this particular candidate was a bad fit.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt.manipulation import (
    AtomInsertion,
    AtomRemoval,
    Manipulation,
    ManipulationLookupError,
    ManipulationRegistry,
    RightGrainTranslation,
    default_registry,
)
from GBOpt.optimization.dispatch import LegacyInvoker, run_legacy_compat_operation
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerValueError,
    OperationSpec,
)

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


def _generic_apply_invoker(operation: Manipulation) -> LegacyInvoker:
    """Return an invoker running ``operation`` through the generic apply() boundary.

    :param operation: Registry-resolved operation, built-in or third-party.
    :return: Invoker matching the shared ``(manipulator, rng) -> (label, atoms)`` shape,
        so it can share retry/selection with the legacy invokers.
    :raises GBMinimizerValueError: If a call produces a child count other than one.
    """

    def _invoke(manipulator: GBManipulator, rng: np.random.Generator):
        del rng  # apply() draws from manipulator's own bound RNG.
        result = manipulator.apply(operation)
        if len(result.children) != 1:
            raise GBMinimizerValueError(
                f"operation {operation.name!r} produced {len(result.children)} "
                "children; MC/GA dispatch requires exactly one"
            )
        (child,) = result.children
        return operation.name, np.array(child.atoms, copy=True)

    return _invoke


class Mutator:
    """Perform randomly selected manipulations on a GB candidate.

    :param choices: Mutation operation names to make available. A name is either one of
        this adapter's three legacy names, or resolved by lookup in ``registry``.
    :param manipulator: GBManipulator used to validate the requested operations.
    :param registry: Keyword argument, optional, defaults to ``None``. Registry used to
        resolve any non-legacy name; ``None`` uses
        ``GBOpt.manipulation.default_registry``.
    """

    # TODO: Add more manipulator options to this class as we make more
    # manipulators faster.

    def __init__(
        self,
        choices: list[str],
        manipulator: GBManipulator,
        *,
        registry: ManipulationRegistry | None = None,
    ):
        self._registry = default_registry if registry is None else registry
        invalid_choices = []
        for name in choices:
            if hasattr(manipulator, name) or name in _LEGACY_OPERATION_TYPES:
                continue
            try:
                operation = self._registry.get(name)
            except ManipulationLookupError:
                invalid_choices.append(name)
                continue
            if operation.arity != 1:
                raise GBMinimizerValueError(
                    f"mutation choice {name!r} resolves to an operation with arity "
                    f"{operation.arity}; only unary (arity 1) operations are usable "
                    "as a mutation choice"
                )
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
        :raises GBMinimizerValueError: If a configured choice cannot be resolved, or a
            registry-resolved operation produces other than one child.
        :raises GBMinimizerError: If no configured mutation can produce a candidate.
        """
        specs: list[OperationSpec] = []
        legacy_invokers: dict[str, LegacyInvoker] = {}
        for name in self.choices_keys:
            if name in _LEGACY_OPERATION_TYPES:
                specs.append(
                    OperationSpec(
                        name=name,
                        operation=_LEGACY_OPERATION_TYPES[name](),
                        weight=1.0,
                    )
                )
                legacy_invokers[name] = _LEGACY_INVOKER_FACTORIES[name](GB)
            else:
                operation = self._registry.get(name)
                specs.append(OperationSpec(name=name, operation=operation, weight=1.0))
                legacy_invokers[name] = _generic_apply_invoker(operation)

        return run_legacy_compat_operation(
            specs,
            rng=local_random,
            manipulator=manipulator,
            legacy_invokers=legacy_invokers,
        )
