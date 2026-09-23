# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.io.types import (
    StructureData,
    StructureFormatError,
    StructureIOError,
    StructureReader,
    StructureValueError,
    StructureWriter,
)


def _atoms(n: int = 2) -> np.ndarray:
    return np.asarray(
        [("Ni", float(i), float(i), float(i)) for i in range(n)],
        dtype=Atom.atom_dtype,
    )


def test_structure_data_accepts_minimal_fields() -> None:
    structure = StructureData(_atoms(3), np.eye(3), np.zeros(3))
    assert structure.atoms.shape == (3,)
    assert np.array_equal(structure.cell, np.eye(3))
    assert np.array_equal(structure.origin, np.zeros(3))
    assert structure.periodicity is None
    assert structure.external_ids is None
    assert structure.charges is None
    assert structure.frame_index is None


def test_structure_data_normalizes_optional_fields() -> None:
    structure = StructureData(
        _atoms(2),
        np.diag([1.0, 2.0, 3.0]),
        [0.0, 0.0, 0.0],
        periodicity=(True, False, np.bool_(True)),
        external_ids=[5, 1],
        charges=[1.5, -1.5],
        frame_index=np.int32(0),
    )
    assert structure.periodicity == (True, False, True)
    assert np.array_equal(structure.external_ids, np.array([5, 1], dtype=np.int64))
    assert structure.external_ids.dtype == np.int64
    assert np.array_equal(structure.charges, np.array([1.5, -1.5]))
    assert structure.frame_index == 0
    assert isinstance(structure.frame_index, int)


def test_structure_data_arrays_are_read_only() -> None:
    structure = StructureData(
        _atoms(1), np.eye(3), np.zeros(3), external_ids=[1], charges=[0.5]
    )
    with pytest.raises(ValueError):
        structure.atoms[0] = ("U", 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        structure.cell[0, 0] = 5.0
    with pytest.raises(ValueError):
        structure.origin[0] = 5.0
    with pytest.raises(ValueError):
        structure.external_ids[0] = 99
    with pytest.raises(ValueError):
        structure.charges[0] = 99.0


def test_structure_data_rejects_wrong_atom_dtype() -> None:
    with pytest.raises(StructureValueError):
        StructureData(np.zeros(3), np.eye(3), np.zeros(3))


def test_structure_data_rejects_non_finite_cell_or_origin() -> None:
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.full((3, 3), np.nan), np.zeros(3))
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), [np.inf, 0.0, 0.0])


def test_structure_data_rejects_wrong_cell_or_origin_shape() -> None:
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(2), np.zeros(3))
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), np.zeros(2))


def test_structure_data_rejects_mismatched_external_ids_or_charges_length() -> None:
    with pytest.raises(StructureValueError):
        StructureData(_atoms(2), np.eye(3), np.zeros(3), external_ids=[1])
    with pytest.raises(StructureValueError):
        StructureData(_atoms(2), np.eye(3), np.zeros(3), charges=[1.0])


def test_structure_data_rejects_non_finite_charges() -> None:
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), np.zeros(3), charges=[np.nan])


def test_structure_data_rejects_malformed_periodicity() -> None:
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), np.zeros(3), periodicity=(True, False))
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), np.zeros(3), periodicity=(1, 0, 0))


def test_structure_data_rejects_negative_or_boolean_frame_index() -> None:
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), np.zeros(3), frame_index=-1)
    with pytest.raises(StructureValueError):
        StructureData(_atoms(1), np.eye(3), np.zeros(3), frame_index=True)


def test_exception_hierarchy_layers_onto_value_error() -> None:
    assert issubclass(StructureFormatError, StructureIOError)
    assert issubclass(StructureFormatError, ValueError)
    assert issubclass(StructureValueError, StructureIOError)
    assert issubclass(StructureValueError, ValueError)


def test_reader_and_writer_protocols_are_structurally_checkable() -> None:
    class _Reader:
        def read(self, path: object, **kwargs: object) -> StructureData:
            return StructureData(_atoms(1), np.eye(3), np.zeros(3))

    class _Writer:
        def write(self, path: object, structure: StructureData, **kwargs: object) -> None:
            return None

    assert isinstance(_Reader(), StructureReader)
    assert isinstance(_Writer(), StructureWriter)
    assert not isinstance(object(), StructureReader)
    assert not isinstance(object(), StructureWriter)
