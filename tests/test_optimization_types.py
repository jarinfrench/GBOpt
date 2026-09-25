# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt._explicit_ownership_evaluation import CandidateEvaluation
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.FileGrainOwnership import CandidateFileMapping, GrainOwnershipError
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
    OperationSpec,
    _CachedEvaluation,
    _candidate_mapping_from_state,
    _candidate_mapping_to_state,
    _FailureDiagnostic,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _successful_evaluation(**overrides) -> CandidateEvaluation:
    kwargs = {
        "candidate_id": "candidate-0",
        "input_index": 0,
        "objective": -1.5,
        "structure_path": "relaxed.data",
        "mapping": _minimal_candidate_mapping(),
        "manipulator": object(),
        "success": True,
        "failure_reason": None,
    }
    kwargs.update(overrides)
    return CandidateEvaluation(**kwargs)


def _failed_evaluation(**overrides) -> CandidateEvaluation:
    kwargs = {
        "candidate_id": "candidate-1",
        "input_index": 2,
        "objective": 1.0e30,
        "structure_path": None,
        "mapping": None,
        "manipulator": None,
        "success": False,
        "failure_reason": "evaluator raised",
    }
    kwargs.update(overrides)
    return CandidateEvaluation(**kwargs)


def _minimal_candidate_mapping() -> CandidateFileMapping:
    return CandidateFileMapping(
        atom_ids=np.asarray([1], dtype=np.int64),
        labels=np.asarray([0], dtype=np.int8),
        species=np.asarray(["Ni"], dtype=object),
        box_dims=np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]], dtype=float),
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )


# --------------------------------------------------------------------------------------
# _CachedEvaluation
# --------------------------------------------------------------------------------------


def test_cached_evaluation_is_immutable():
    cached = _CachedEvaluation(energy=-2.0, structure_path="dump.data")

    assert cached.energy == -2.0
    assert cached.structure_path == "dump.data"
    with pytest.raises(AttributeError):
        cached.energy = -3.0  # ty: ignore[invalid-assignment]


# --------------------------------------------------------------------------------------
# _FailureDiagnostic.from_evaluation
# --------------------------------------------------------------------------------------


def test_failure_diagnostic_from_evaluation_captures_failure_context():
    record = _failed_evaluation(failure_reason="LAMMPS crashed")

    diagnostic = _FailureDiagnostic.from_evaluation(record, generation=3)

    assert diagnostic.candidate_id == "candidate-1"
    assert diagnostic.generation == 3
    assert diagnostic.input_index == 2
    assert diagnostic.failure_reason == "LAMMPS crashed"
    assert diagnostic.source_path is None


def test_failure_diagnostic_from_evaluation_rejects_successful_record():
    record = _successful_evaluation()

    with pytest.raises(GBMinimizerValueError, match="failed CandidateEvaluation"):
        _FailureDiagnostic.from_evaluation(record, generation=0)


@pytest.mark.parametrize("generation", [-1, 1.5, True])
def test_failure_diagnostic_from_evaluation_rejects_invalid_generation(generation):
    record = _failed_evaluation()

    with pytest.raises(GBMinimizerValueError, match="non-negative integer"):
        _FailureDiagnostic.from_evaluation(record, generation=generation)


# --------------------------------------------------------------------------------------
# _FailureDiagnostic.to_state / from_state
# --------------------------------------------------------------------------------------


def test_failure_diagnostic_state_round_trips():
    diagnostic = _FailureDiagnostic(
        candidate_id="candidate-2",
        generation=4,
        input_index=1,
        failure_reason="reconstruction failed",
        source_path="failed.data",
    )

    restored = _FailureDiagnostic.from_state(diagnostic.to_state())

    assert restored == diagnostic


def test_failure_diagnostic_from_state_rejects_non_dict():
    with pytest.raises(GBMinimizerError, match="must be a dictionary"):
        _FailureDiagnostic.from_state("not-a-dict")


def test_failure_diagnostic_from_state_rejects_incomplete_state():
    with pytest.raises(GBMinimizerError, match="incomplete"):
        _FailureDiagnostic.from_state({"candidate_id": "x"})


def test_failure_diagnostic_from_state_rejects_negative_generation():
    state = {
        "candidate_id": "candidate-3",
        "generation": -1,
        "input_index": 0,
        "failure_reason": "bad",
        "source_path": None,
    }

    with pytest.raises(GBMinimizerError, match="non-negative integer"):
        _FailureDiagnostic.from_state(state)


def test_failure_diagnostic_from_state_rejects_blank_failure_reason():
    state = {
        "candidate_id": "candidate-4",
        "generation": 0,
        "input_index": 0,
        "failure_reason": "",
        "source_path": None,
    }

    with pytest.raises(GBMinimizerError, match="non-empty string"):
        _FailureDiagnostic.from_state(state)


# --------------------------------------------------------------------------------------
# _candidate_mapping_to_state / _candidate_mapping_from_state
# --------------------------------------------------------------------------------------


def test_candidate_mapping_state_round_trips():
    mapping = _minimal_candidate_mapping()

    restored = _candidate_mapping_from_state(_candidate_mapping_to_state(mapping))

    assert restored.atom_ids.tolist() == mapping.atom_ids.tolist()
    assert restored.labels.tolist() == mapping.labels.tolist()
    assert restored.species.tolist() == mapping.species.tolist()
    assert restored.gb_plane_x == mapping.gb_plane_x
    assert restored.inplane_periodic == mapping.inplane_periodic
    assert restored.coordinate_tolerance == mapping.coordinate_tolerance
    assert restored.normal_topology == mapping.normal_topology


def test_candidate_mapping_from_state_rejects_non_dict():
    with pytest.raises(GrainOwnershipError, match="must be a dictionary"):
        _candidate_mapping_from_state("not-a-dict")


def test_candidate_mapping_from_state_rejects_incomplete_state():
    incomplete_state = _candidate_mapping_to_state(_minimal_candidate_mapping())
    del incomplete_state["gb_plane_x"]

    with pytest.raises(GrainOwnershipError, match="incomplete or malformed"):
        _candidate_mapping_from_state(incomplete_state)


# --------------------------------------------------------------------------------------
# OperationSpec
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
        return None


def test_operation_spec_exposes_name_operation_weight_and_arity():
    operation = _StubOperation(arity=2, name="stub")
    spec = OperationSpec(name="stub", operation=operation, weight=1.5)
    assert spec.name == "stub"
    assert spec.operation is operation
    assert spec.weight == 1.5
    assert spec.arity == 2


def test_operation_spec_default_param_sampler_returns_no_params():
    spec = OperationSpec(name="stub", operation=_StubOperation(), weight=1.0)
    assert dict(spec.param_sampler(np.random.default_rng(0), ())) == {}


def test_operation_spec_custom_param_sampler_is_used():
    def sampler(rng, parents):
        return {"amount": 3.0}

    spec = OperationSpec(
        name="stub", operation=_StubOperation(), weight=1.0, param_sampler=sampler
    )
    assert dict(spec.param_sampler(np.random.default_rng(0), ())) == {"amount": 3.0}


def test_operation_spec_rejects_non_string_name():
    with pytest.raises(GBMinimizerTypeError):
        OperationSpec(name=1, operation=_StubOperation(), weight=1.0)


def test_operation_spec_rejects_empty_name():
    with pytest.raises(GBMinimizerValueError):
        OperationSpec(name="", operation=_StubOperation(), weight=1.0)


def test_operation_spec_rejects_operation_missing_the_protocol():
    with pytest.raises(GBMinimizerTypeError):
        OperationSpec(name="stub", operation=object(), weight=1.0)


def test_operation_spec_rejects_non_positive_arity():
    with pytest.raises(GBMinimizerValueError):
        OperationSpec(name="stub", operation=_StubOperation(arity=0), weight=1.0)


def test_operation_spec_rejects_boolean_weight():
    with pytest.raises(GBMinimizerTypeError):
        OperationSpec(name="stub", operation=_StubOperation(), weight=True)


def test_operation_spec_rejects_negative_weight():
    with pytest.raises(GBMinimizerValueError):
        OperationSpec(name="stub", operation=_StubOperation(), weight=-1.0)


def test_operation_spec_rejects_non_finite_weight():
    with pytest.raises(GBMinimizerValueError):
        OperationSpec(name="stub", operation=_StubOperation(), weight=float("inf"))


def test_operation_spec_rejects_uncallable_param_sampler():
    with pytest.raises(GBMinimizerTypeError):
        OperationSpec(
            name="stub", operation=_StubOperation(), weight=1.0, param_sampler="nope"
        )


def test_operation_spec_accepts_zero_weight():
    spec = OperationSpec(name="stub", operation=_StubOperation(), weight=0.0)
    assert spec.weight == 0.0


def test_operation_spec_is_immutable():
    spec = OperationSpec(name="stub", operation=_StubOperation(), weight=1.0)
    with pytest.raises(AttributeError):
        spec.weight = 2.0
