# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.builtins: explicit built-in operation registration."""

import pytest

from GBOpt.manipulation.builtins import register_builtin_operations
from GBOpt.manipulation.crossover import SliceAndMerge
from GBOpt.manipulation.density import AtomInsertion, AtomRemoval
from GBOpt.manipulation.registry import ManipulationRegistry
from GBOpt.manipulation.separation import InterfaceSeparation
from GBOpt.manipulation.soft_mode import SoftModeDisplacement
from GBOpt.manipulation.termination import GrainTerminationCycle
from GBOpt.manipulation.translation import RightGrainTranslation
from GBOpt.manipulation.types import (
    ManipulationContext,
    ManipulationRegistrationError,
    ManipulationResult,
)

_EXPECTED_BUILTINS = {
    "right_grain_translation": RightGrainTranslation,
    "grain_termination_cycle": GrainTerminationCycle,
    "interface_separation": InterfaceSeparation,
    "atom_insertion": AtomInsertion,
    "atom_removal": AtomRemoval,
    "soft_mode_displacement": SoftModeDisplacement,
    "slice_and_merge": SliceAndMerge,
}


def test_registers_every_builtin_under_its_own_name():
    registry = ManipulationRegistry()
    register_builtin_operations(registry)
    for name, operation_type in _EXPECTED_BUILTINS.items():
        assert isinstance(registry.get(name), operation_type)


def test_calling_twice_against_the_same_registry_is_idempotent():
    registry = ManipulationRegistry()
    register_builtin_operations(registry)
    register_builtin_operations(registry)
    for name, operation_type in _EXPECTED_BUILTINS.items():
        assert isinstance(registry.get(name), operation_type)


def test_conflicting_registration_under_a_builtin_name_raises():
    class _StubManipulation:
        @property
        def name(self) -> str:
            return "slice_and_merge"

        @property
        def arity(self) -> int:
            return 2

        def execute(self, context: ManipulationContext) -> ManipulationResult:
            return ManipulationResult(children=context.parents)

    registry = ManipulationRegistry()
    registry.register("slice_and_merge", _StubManipulation())
    with pytest.raises(ManipulationRegistrationError):
        register_builtin_operations(registry)


def test_default_registry_target_registers_into_the_shared_registry():
    from GBOpt.manipulation.registry import default_registry

    register_builtin_operations()
    assert isinstance(default_registry.get("slice_and_merge"), SliceAndMerge)
