# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.GBManipulator import GBManipulatorValueError
from GBOpt.optimization.mutation import GBMinimizerError, GBMinimizerValueError, Mutator

# --------------------------------------------------------------------------------------
# Mutator.__init__
# --------------------------------------------------------------------------------------


def test_mutator_rejects_unknown_choices():
    class Manipulator:
        def translate_right_grain(self, *, dy, dz):
            raise NotImplementedError

    with pytest.raises(
        GBMinimizerValueError,
        match=r"Unknown GBManipulator mutation choice\(s\).*not_a_real_operation",
    ):
        Mutator(["translate_right_grain", "not_a_real_operation"], Manipulator())


def test_mutator_rejects_empty_choices():
    class Manipulator:
        pass

    with pytest.raises(
        GBMinimizerValueError, match="At least one mutation choice must be provided"
    ):
        Mutator([], Manipulator())


# --------------------------------------------------------------------------------------
# Mutator.mutate
# --------------------------------------------------------------------------------------


def test_translation_mutation_uses_current_parent_dimensions():
    class FixedRandom:
        def permutation(self, size):
            assert size == 1
            return np.array([0])

        def uniform(self, _lower, _upper):
            return 0.5

    class ParentStub:
        box_dims = np.asarray(
            [[0.0, 10.0], [-2.0, 18.0], [5.0, 35.0]],
            dtype=float,
        )

    class ManipulatorStub:
        def __init__(self):
            self.parents = [ParentStub()]
            self.translation = None

        def translate_right_grain(self, *, dy, dz):
            self.translation = (dy, dz)
            return np.empty(0)

    class GBStub:
        repeat_factor = (2, 5)
        # Deliberately unrelated to the current parent so the regression fails if the
        # mutation falls back to reference GBMaker dimensions.
        y_dim = 1_000.0
        z_dim = 2_000.0

    manipulator = ManipulatorStub()
    mutator = Mutator(["translate_right_grain"], manipulator)
    mutation, _candidate = mutator.mutate(
        FixedRandom(),
        GBStub(),
        manipulator,
    )

    # Current y/z lengths are 20 and 30 A. At a fixed random fraction of 0.5, repeat
    # factors (2, 5) therefore give dy=5 A and dz=3 A.
    assert manipulator.translation == pytest.approx((5.0, 3.0))
    assert mutation == "shift5.00000000dy3.00000000dz"


def test_mutator_retries_after_infeasible_mutation():
    """An infeasible mutation should fall through to another configured choice."""

    class FakeRandom:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

        def uniform(self, low, high):
            assert low == 0
            assert high == 1
            return 0.5

    class Parent:
        box_dims = np.array(
            [
                [0.0, 10.0],
                [0.0, 20.0],
                [0.0, 30.0],
            ]
        )

    class Manipulator:
        parents = [Parent()]

        def __init__(self):
            self.calls = []

        def remove_atoms(self, *, num_to_remove):
            self.calls.append("remove_atoms")
            assert num_to_remove == 1
            raise GBManipulatorValueError(
                "Not enough neighbor atoms of type 2 to remove."
            )

        def translate_right_grain(self, *, dy, dz):
            self.calls.append("translate_right_grain")
            np.testing.assert_allclose(dy, 10.0)
            np.testing.assert_allclose(dz, 15.0)
            return "translated-system"

    class GB:
        repeat_factor = (1, 1)

    manipulator = Manipulator()
    mutator = Mutator(
        ["remove_atoms", "translate_right_grain"],
        manipulator,
    )

    mutation, new_system = mutator.mutate(
        local_random=FakeRandom(),
        GB=GB(),
        manipulator=manipulator,
    )

    assert manipulator.calls == [
        "remove_atoms",
        "translate_right_grain",
    ]
    assert mutation == "shift10.00000000dy15.00000000dz"
    assert new_system == "translated-system"


def test_mutator_does_not_hide_unexpected_mutation_error():
    """Only expected mutation-infeasibility errors should trigger a retry."""

    class FakeRandom:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

    class Manipulator:
        def __init__(self):
            self.calls = []

        def remove_atoms(self, *, num_to_remove):
            self.calls.append("remove_atoms")
            assert num_to_remove == 1
            raise RuntimeError("unexpected mutation failure")

        def translate_right_grain(self, *, dy, dz):
            self.calls.append("translate_right_grain")
            return "translated-system"

    class GB:
        repeat_factor = (1, 1)

    manipulator = Manipulator()
    mutator = Mutator(
        ["remove_atoms", "translate_right_grain"],
        manipulator,
    )

    with pytest.raises(RuntimeError, match="unexpected mutation failure"):
        mutator.mutate(
            local_random=FakeRandom(),
            GB=GB(),
            manipulator=manipulator,
        )

    assert manipulator.calls == ["remove_atoms"]


def test_mutator_fails_when_all_mutations_are_infeasible():
    """The optimizer should fail clearly when every configured mutation is infeasible."""

    class FakeRandom:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

        def uniform(self, low, high):
            assert low == 0
            assert high == 1
            return 0.5

    class Parent:
        box_dims = np.array(
            [
                [0.0, 10.0],
                [0.0, 20.0],
                [0.0, 30.0],
            ]
        )

    class Manipulator:
        parents = [Parent()]

        def __init__(self):
            self.calls = []

        def remove_atoms(self, *, num_to_remove):
            self.calls.append("remove_atoms")
            assert num_to_remove == 1
            raise GBManipulatorValueError("removal is infeasible")

        def translate_right_grain(self, *, dy, dz):
            self.calls.append("translate_right_grain")
            raise GBManipulatorValueError("translation is infeasible")

    class GB:
        repeat_factor = (1, 1)

    manipulator = Manipulator()
    mutator = Mutator(
        ["remove_atoms", "translate_right_grain"],
        manipulator,
    )

    with pytest.raises(
        GBMinimizerError,
        match="No configured mutation could produce a valid candidate",
    ) as exc_info:
        mutator.mutate(
            local_random=FakeRandom(),
            GB=GB(),
            manipulator=manipulator,
        )

    message = str(exc_info.value)
    assert "remove_atoms" in message
    assert "removal is infeasible" in message
    assert "translate_right_grain" in message
    assert "translation is infeasible" in message
    assert manipulator.calls == [
        "remove_atoms",
        "translate_right_grain",
    ]
    assert isinstance(exc_info.value.__cause__, GBManipulatorValueError)
