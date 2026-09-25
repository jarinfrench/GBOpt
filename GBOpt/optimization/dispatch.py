# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""OperationSpec-driven operation selection, replacing hard-coded name dispatch.

Two execution shapes coexist here, both selected by the same weighted-order-with-retry
algorithm operating on ``OperationSpec.weight``/``arity``:

- Legacy-compatibility execution (``run_legacy_compat_operation``): invokes one of
  ``GBManipulator``'s own established per-operation methods directly, via a small
  name-keyed table of invokers, bypassing ``ManipulationContext``/``InterfaceCandidate``
  entirely. This exists because at least one legacy method's own established tolerance
  (``translate_right_grain`` accepting a relaxed right-grain atom that has legitimately
  crossed the nominal interface plane) is stricter than ``InterfaceCandidate.__init__``
  allows, and MC/GA reload every manipulator from a relaxed evaluator output, so this
  is not a theoretical edge case. Preserving today's fixed-seed histories for the
  ``choices: list[str]`` compatibility adapter requires calling the same methods the
  same way, not routing through the newer, stricter boundary.
- Generic execution (elsewhere, via ``GBManipulator.apply()`` or direct
  ``Manipulation.execute()``): used for any operation reached outside the legacy
  compatibility adapter (a built-in used directly, or a third-party operation), which
  has no legacy tolerance to preserve and so can safely use the stricter, uniform
  ``InterfaceCandidate``-based boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from GBOpt.GBManipulator import GBManipulator, GBManipulatorValueError
from GBOpt.optimization.types import GBMinimizerError, OperationSpec

LegacyInvoker = Callable[[GBManipulator, np.random.Generator], tuple[str, np.ndarray]]


def select_operation_order(
    rng: np.random.Generator, specs: Sequence[OperationSpec]
) -> np.ndarray:
    """Return an order in which to try ``specs``, weighted without replacement.

    When every spec shares the same weight (the only case exercised by any existing
    default configuration), this reduces to exactly ``rng.permutation(len(specs))`` --
    the same call, consuming RNG state identically, that the pre-OperationSpec mutation
    dispatch already made. A genuinely non-uniform weighting (only reachable through a
    caller-supplied ``OperationSpec`` list, never through the legacy compatibility
    adapter) uses the Efraimidis-Spirakis weighted-reservoir ordering instead.

    :param rng: Random-number generator to draw from.
    :param specs: Nonempty specs to order.
    :return: Array of ``specs`` indices, in the order to try them.
    """
    weights = np.array([spec.weight for spec in specs], dtype=float)
    if np.all(weights == weights[0]):
        return rng.permutation(len(specs))
    # Efraimidis-Spirakis: draw one key per item and sort descending. A zero-weight
    # item's key is always 0.0 (never selected ahead of a positive-weight item, sorted
    # last among ties by construction since a duplicate 0.0 falls back to array order
    # under a stable sort).
    keys = np.zeros(len(specs), dtype=float)
    positive = weights > 0.0
    keys[positive] = rng.random(int(np.count_nonzero(positive))) ** (
        1.0 / weights[positive]
    )
    return np.argsort(-keys, kind="stable")


def run_legacy_compat_operation(
    specs: Sequence[OperationSpec],
    *,
    rng: np.random.Generator,
    manipulator: GBManipulator,
    legacy_invokers: Mapping[str, LegacyInvoker],
) -> tuple[str, np.ndarray]:
    """Try ``specs`` in weighted order until one legacy invocation succeeds.

    Mirrors the pre-OperationSpec ``Mutator.mutate``/``_apply_mutation`` shape exactly:
    each configured operation is attempted at most once, in the same weighted (by
    default, uniform-permutation) order; an operation whose invocation raises
    ``GBManipulatorValueError`` is infeasible for the current candidate and the next
    configured operation is tried; failure of every configured operation is fatal.

    :param specs: Keyword argument omitted; nonempty legacy-compatibility specs to try.
    :param rng: Keyword argument, required. Random-number generator to draw from.
    :param manipulator: Keyword argument, required. Manipulator to operate on.
    :param legacy_invokers: Keyword argument, required. Maps each spec's ``name`` to a
        callable performing that operation directly against ``manipulator``.
    :return: The successful operation's label and resulting atom positions.
    :raises GBMinimizerError: If no configured operation can produce a valid candidate.
    """
    order = select_operation_order(rng, specs)
    failures: list[tuple[str, GBManipulatorValueError]] = []
    for index in order:
        spec = specs[int(index)]
        try:
            return legacy_invokers[spec.name](manipulator, rng)
        except GBManipulatorValueError as exc:
            failures.append((spec.name, exc))
    failure_details = "; ".join(f"{name}: {exc}" for name, exc in failures)
    raise GBMinimizerError(
        "No configured mutation could produce a valid candidate. Attempted "
        f"mutations: {failure_details}"
    ) from failures[-1][1]


__all__ = [
    "LegacyInvoker",
    "select_operation_order",
    "run_legacy_compat_operation",
]
