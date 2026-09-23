# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import numpy as np
import pytest

from GBOpt.io.lammps.tokens import (
    _INT64_MAX,
    _normalize_type_mapping,
    _strict_id_token,
    _strict_species_name,
    _strict_type_id,
)
from GBOpt.io.lammps.types import LammpsDataError


def test_strict_id_token_accepts_positive_integral_token() -> None:
    assert _strict_id_token("42") == 42


@pytest.mark.parametrize("token", ["0", "-1", "1.5", "abc", "1_000", ""])
def test_strict_id_token_rejects_nonpositive_or_nonintegral_tokens(token: str) -> None:
    with pytest.raises(LammpsDataError):
        _strict_id_token(token)


def test_strict_id_token_rejects_values_beyond_int64_range() -> None:
    with pytest.raises(LammpsDataError):
        _strict_id_token(str(_INT64_MAX + 1))


def test_strict_species_name_accepts_supported_symbol() -> None:
    assert _strict_species_name("Ni") == "Ni"


@pytest.mark.parametrize("value", ["Xx", 1, None, ""])
def test_strict_species_name_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(LammpsDataError):
        _strict_species_name(value)


def test_strict_type_id_accepts_positive_integer() -> None:
    assert _strict_type_id(3) == 3
    assert _strict_type_id(np.int64(3)) == 3


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1"])
def test_strict_type_id_rejects_invalid_values(value: object) -> None:
    with pytest.raises(LammpsDataError):
        _strict_type_id(value)


def test_normalize_type_mapping_accepts_none_or_empty() -> None:
    assert _normalize_type_mapping(None) == {}
    assert _normalize_type_mapping({}) == {}


def test_normalize_type_mapping_accepts_name_to_id_form() -> None:
    assert _normalize_type_mapping({"Ni": 1, "O": 2}) == {1: "Ni", 2: "O"}


def test_normalize_type_mapping_accepts_id_to_name_form() -> None:
    assert _normalize_type_mapping({1: "Ni", 2: "O"}) == {1: "Ni", 2: "O"}


def test_normalize_type_mapping_rejects_ambiguous_reverse_mapping() -> None:
    with pytest.raises(LammpsDataError):
        _normalize_type_mapping({"Ni": 1, "O": 1})


@pytest.mark.parametrize("value", ["not a mapping", 5, ["Ni", 1]])
def test_normalize_type_mapping_rejects_non_mapping(value: object) -> None:
    with pytest.raises(LammpsDataError):
        _normalize_type_mapping(value)


def test_normalize_type_mapping_rejects_mixed_key_forms() -> None:
    with pytest.raises(LammpsDataError):
        _normalize_type_mapping({"Ni": 1, 2: "O"})
