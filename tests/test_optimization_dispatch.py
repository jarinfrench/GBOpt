# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.optimization.dispatch: OperationSpec-driven selection and retry."""

import numpy as np
import pytest

from GBOpt.GBManipulator import GBManipulatorValueError
from GBOpt.optimization.dispatch import run_legacy_compat_operation, select_operation_order
from GBOpt.optimization.types import GBMinimizerError, OperationSpec

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class _StubOperation:
    def __init__(self, *, arity: int = 1, name: str = "stub") -> None:
        self._arity = arity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return self._arity

    def execute(self, context):
        raise AssertionError("execute must not run for legacy-compat dispatch")


def _spec(name: str, weight: float = 1.0) -> OperationSpec:
    return OperationSpec(name=name, operation=_StubOperation(name=name), weight=weight)


# --------------------------------------------------------------------------------------
# select_operation_order
# --------------------------------------------------------------------------------------


def test_equal_weights_delegate_to_a_single_permutation_call():
    class FixedRandom:
        def permutation(self, size):
            assert size == 3
            return np.array([2, 0, 1])

    specs = [_spec("a"), _spec("b"), _spec("c")]
    order = select_operation_order(FixedRandom(), specs)
    np.testing.assert_array_equal(order, [2, 0, 1])


def test_unequal_weights_favor_the_higher_weight_spec_over_many_trials():
    rng = np.random.default_rng(0)
    specs = [_spec("low", weight=1.0), _spec("high", weight=99.0)]
    first_choice_counts = {"low": 0, "high": 0}
    for _ in range(500):
        order = select_operation_order(rng, specs)
        first_choice_counts[specs[int(order[0])].name] += 1
    assert first_choice_counts["high"] > first_choice_counts["low"]


def test_zero_weight_spec_is_never_selected_first_when_a_positive_weight_exists():
    rng = np.random.default_rng(1)
    specs = [_spec("never", weight=0.0), _spec("always", weight=1.0)]
    for _ in range(50):
        order = select_operation_order(rng, specs)
        assert specs[int(order[0])].name == "always"


def test_weighted_order_is_a_permutation_of_all_indices():
    rng = np.random.default_rng(2)
    specs = [_spec("a", weight=1.0), _spec("b", weight=5.0), _spec("c", weight=2.0)]
    order = select_operation_order(rng, specs)
    assert sorted(int(i) for i in order) == [0, 1, 2]


# --------------------------------------------------------------------------------------
# run_legacy_compat_operation
# --------------------------------------------------------------------------------------


def test_runs_the_first_spec_in_order_that_succeeds():
    class FixedRandom:
        def permutation(self, size):
            return np.array(range(size))

    specs = [_spec("a"), _spec("b")]
    invokers = {
        "a": lambda manipulator, rng: ("a-result", "atoms-a"),
        "b": lambda manipulator, rng: pytest.fail("b should not run"),
    }
    label, atoms = run_legacy_compat_operation(
        specs, rng=FixedRandom(), manipulator=object(), legacy_invokers=invokers
    )
    assert (label, atoms) == ("a-result", "atoms-a")


def test_retries_the_next_spec_after_an_infeasible_one():
    class FixedRandom:
        def permutation(self, size):
            return np.array(range(size))

    calls = []

    def _fail(manipulator, rng):
        calls.append("a")
        raise GBManipulatorValueError("infeasible")

    def _succeed(manipulator, rng):
        calls.append("b")
        return "b-result", "atoms-b"

    specs = [_spec("a"), _spec("b")]
    invokers = {"a": _fail, "b": _succeed}
    label, atoms = run_legacy_compat_operation(
        specs, rng=FixedRandom(), manipulator=object(), legacy_invokers=invokers
    )
    assert calls == ["a", "b"]
    assert (label, atoms) == ("b-result", "atoms-b")


def test_raises_when_every_spec_is_infeasible():
    class FixedRandom:
        def permutation(self, size):
            return np.array(range(size))

    specs = [_spec("a"), _spec("b")]
    invokers = {
        "a": lambda manipulator, rng: (_ for _ in ()).throw(
            GBManipulatorValueError("a is infeasible")
        ),
        "b": lambda manipulator, rng: (_ for _ in ()).throw(
            GBManipulatorValueError("b is infeasible")
        ),
    }
    with pytest.raises(
        GBMinimizerError, match="No configured mutation could produce a valid candidate"
    ) as exc_info:
        run_legacy_compat_operation(
            specs, rng=FixedRandom(), manipulator=object(), legacy_invokers=invokers
        )
    message = str(exc_info.value)
    assert "a: a is infeasible" in message
    assert "b: b is infeasible" in message
    assert isinstance(exc_info.value.__cause__, GBManipulatorValueError)


def test_does_not_catch_an_unexpected_exception_type():
    class FixedRandom:
        def permutation(self, size):
            return np.array(range(size))

    specs = [_spec("a")]
    invokers = {
        "a": lambda manipulator, rng: (_ for _ in ()).throw(RuntimeError("boom")),
    }
    with pytest.raises(RuntimeError, match="boom"):
        run_legacy_compat_operation(
            specs, rng=FixedRandom(), manipulator=object(), legacy_invokers=invokers
        )
