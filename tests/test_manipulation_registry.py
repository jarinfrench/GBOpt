# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.registry: explicit registration and lookup."""

import pytest

from GBOpt.manipulation.registry import ManipulationRegistry
from GBOpt.manipulation.types import (
    ManipulationContext,
    ManipulationLookupError,
    ManipulationRegistrationError,
    ManipulationResult,
)


class _StubManipulation:
    def __init__(self, *, arity: int = 1, name: str = "stub") -> None:
        self._arity = arity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return self._arity

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        return ManipulationResult(children=context.parents)


def test_register_then_get_returns_the_same_operation():
    registry = ManipulationRegistry()
    operation = _StubManipulation()
    registry.register("stub", operation)
    assert registry.get("stub") is operation


def test_register_duplicate_name_raises():
    registry = ManipulationRegistry()
    registry.register("stub", _StubManipulation())
    with pytest.raises(ManipulationRegistrationError):
        registry.register("stub", _StubManipulation())


def test_get_unknown_name_raises():
    registry = ManipulationRegistry()
    with pytest.raises(ManipulationLookupError):
        registry.get("missing")


def test_a_fresh_registry_has_no_import_time_registrations():
    registry = ManipulationRegistry()
    with pytest.raises(ManipulationLookupError):
        registry.get("anything")


def test_registries_are_independent_of_each_other():
    first = ManipulationRegistry()
    second = ManipulationRegistry()
    first.register("stub", _StubManipulation())
    with pytest.raises(ManipulationLookupError):
        second.get("stub")


def test_module_level_default_registry_starts_empty():
    from GBOpt.manipulation.registry import default_registry

    with pytest.raises(ManipulationLookupError):
        default_registry.get("nothing-should-be-registered-here")
