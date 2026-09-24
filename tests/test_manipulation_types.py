# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.types: context/result value types and the protocol."""

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.manipulation.types import (
    Manipulation,
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationResult,
)

_BOX_DIMS = np.asarray([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]], dtype=float)
_GB_PLANE_X = 10.0
_LEFT_BOUNDS = (0.0, 9.5)
_RIGHT_BOUNDS = (10.5, 20.0)
_TOLERANCE = 1.0e-8


def _atoms() -> np.ndarray:
    return np.asarray(
        [
            ("U", 3.0, 1.0, 1.0),
            ("O", 7.5, 2.0, 3.0),
            ("O", 12.5, 4.0, 5.0),
            ("U", 17.0, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )


def _labels() -> np.ndarray:
    return np.asarray(
        [LEFT_GRAIN_LABEL, LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
        dtype=np.int8,
    )


def _make_candidate(**overrides) -> InterfaceCandidate:
    kwargs = {
        "atoms": _atoms(),
        "box_dims": _BOX_DIMS,
        "gb_plane_x": _GB_PLANE_X,
        "left_grain_x_bounds": _LEFT_BOUNDS,
        "right_grain_x_bounds": _RIGHT_BOUNDS,
        "grain_labels": _labels(),
        "inplane_periodic": (True, True),
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        "coordinate_tolerance": _TOLERANCE,
        "interface_separation": 0.0,
    }
    kwargs.update(overrides)
    return InterfaceCandidate(**kwargs)


class _EchoManipulation:
    """Minimal external (non-GBOpt) operation satisfying the Manipulation protocol."""

    def __init__(self, *, arity: int = 1, name: str = "echo") -> None:
        self._arity = arity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return self._arity

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        return ManipulationResult(
            children=context.parents,
            parameters=dict(context.params),
            lineage={"name": self.name, "parent_count": len(context.parents)},
        )


# --------------------------------------------------------------------------------------
# ManipulationContext
# --------------------------------------------------------------------------------------


def test_context_accepts_one_or_more_parents():
    rng = np.random.default_rng(0)
    context = ManipulationContext(parents=(_make_candidate(),), rng=rng)
    assert context.parents == (context.parents[0],)
    assert context.rng is rng
    assert dict(context.params) == {}


def test_context_stores_two_parents_in_order():
    rng = np.random.default_rng(0)
    first = _make_candidate()
    second = _make_candidate(interface_separation=1.0)
    context = ManipulationContext(parents=(first, second), rng=rng)
    assert context.parents == (first, second)


def test_context_rejects_empty_parents():
    with pytest.raises(ManipulationConfigurationError):
        ManipulationContext(parents=(), rng=np.random.default_rng(0))


def test_context_rejects_non_candidate_parent():
    with pytest.raises(ManipulationConfigurationError):
        ManipulationContext(parents=(object(),), rng=np.random.default_rng(0))


def test_context_rejects_non_generator_rng():
    with pytest.raises(ManipulationConfigurationError):
        ManipulationContext(parents=(_make_candidate(),), rng=0)


def test_context_params_defaults_to_empty_and_is_read_only():
    context = ManipulationContext(
        parents=(_make_candidate(),), rng=np.random.default_rng(0)
    )
    with pytest.raises(TypeError):
        context.params["x"] = 1  # type: ignore[index]


def test_context_params_are_independent_of_the_caller_dict():
    source = {"a": 1}
    context = ManipulationContext(
        parents=(_make_candidate(),), rng=np.random.default_rng(0), params=source
    )
    source["a"] = 2
    assert context.params["a"] == 1


def test_context_is_immutable():
    context = ManipulationContext(
        parents=(_make_candidate(),), rng=np.random.default_rng(0)
    )
    with pytest.raises(AttributeError):
        context.rng = np.random.default_rng(1)  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# ManipulationResult
# --------------------------------------------------------------------------------------


def test_result_stores_children_parameters_and_lineage():
    child = _make_candidate()
    result = ManipulationResult(
        children=(child,), parameters={"amount": 1.0}, lineage={"name": "echo"}
    )
    assert result.children == (child,)
    assert dict(result.parameters) == {"amount": 1.0}
    assert dict(result.lineage) == {"name": "echo"}


def test_result_rejects_empty_children():
    with pytest.raises(ManipulationConfigurationError):
        ManipulationResult(children=())


def test_result_rejects_non_candidate_child():
    with pytest.raises(ManipulationConfigurationError):
        ManipulationResult(children=(object(),))


def test_result_parameters_and_lineage_default_to_empty_and_are_read_only():
    result = ManipulationResult(children=(_make_candidate(),))
    assert dict(result.parameters) == {}
    assert dict(result.lineage) == {}
    with pytest.raises(TypeError):
        result.parameters["x"] = 1  # type: ignore[index]


def test_result_is_immutable():
    result = ManipulationResult(children=(_make_candidate(),))
    with pytest.raises(AttributeError):
        result.children = ()  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Manipulation protocol
# --------------------------------------------------------------------------------------


def test_external_operation_satisfies_the_manipulation_protocol():
    operation = _EchoManipulation()
    assert isinstance(operation, Manipulation)


def test_external_operation_executes_against_a_context():
    operation = _EchoManipulation(arity=1, name="echo")
    candidate = _make_candidate()
    context = ManipulationContext(
        parents=(candidate,), rng=np.random.default_rng(0), params={"amount": 2.0}
    )
    result = operation.execute(context)
    assert result.children == (candidate,)
    assert dict(result.parameters) == {"amount": 2.0}
    assert dict(result.lineage) == {"name": "echo", "parent_count": 1}
