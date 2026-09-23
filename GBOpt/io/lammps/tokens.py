# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared LAMMPS token-level parsing helpers for ``GBOpt.io.lammps``.

Contains the atom-ID, species, and atom-type token validators and the type-mapping
normalizer shared by ``GBOpt.io.lammps.data`` and ``GBOpt.io.lammps.dump``. This module
is a leaf with respect to its ``io.lammps`` siblings: it depends only on ``types``
(for ``LammpsDataError``) and ``GBOpt.Atom``, and neither ``data`` nor ``dump`` is
imported here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.io.lammps.types import LammpsDataError

_INTEGER_TOKEN = re.compile(r"^[+-]?\d+$")
_INT64_MAX = np.iinfo(np.int64).max


def _strict_id_token(token: str) -> int:
    """Validate one positive file atom-ID token.

    :param token: Raw atom-ID token read from a LAMMPS file.
    :return: A positive Python integer representable as signed ``int64``.
    :raises LammpsDataError: If the token is nonintegral, nonpositive, or outside the
        supported ``int64`` range.
    """
    if not _INTEGER_TOKEN.fullmatch(token):
        raise LammpsDataError(f"atom ID must be an integral token, got {token!r}")

    value = int(token)
    if value <= 0:
        raise LammpsDataError("atom IDs must be positive")
    if value > _INT64_MAX:
        raise LammpsDataError(
            "atom ID must be representable as a signed 64-bit integer"
        )
    return value


def _strict_species_name(value: object) -> str:
    """Validate one GBOpt element symbol.

    :param value: Species value to validate.
    :return: The validated element symbol.
    :raises LammpsDataError: If the value is not a supported element symbol.
    """
    if not isinstance(value, str) or value not in Atom._numbers:
        raise LammpsDataError(f"unsupported atom species label: {value!r}")
    return value


def _resolve_type_id_species(
    type_id: int,
    id_to_name: Mapping[int, str],
    inverse_default: Mapping[int, str],
) -> str:
    """Resolve a numeric atom-type ID to a validated element symbol.

    Shared by the data and dump readers' per-row species resolution, which otherwise
    differ only in the file-format-specific checks each performs before calling this.

    :param type_id: Positive atom-type ID token from a LAMMPS data or dump row.
    :param id_to_name: Caller-supplied type-ID-to-species mapping, normalized by
        ``_normalize_type_mapping``. Checked first when non-empty.
    :param inverse_default: Fallback atomic-number-to-symbol mapping, used only when
        ``id_to_name`` is empty.
    :return: The validated element symbol.
    :raises LammpsDataError: If ``type_id`` is not found in the applicable mapping, or
        the resolved species is unsupported.
    """
    if id_to_name:
        try:
            species = id_to_name[type_id]
        except KeyError as exc:
            raise LammpsDataError(
                f"type id {type_id} not found in type mapping"
            ) from exc
    else:
        try:
            species = inverse_default[type_id]
        except KeyError as exc:
            raise LammpsDataError(f"unknown atom type id {type_id}") from exc
    return _strict_species_name(species)


def _strict_type_id(value: object) -> int:
    """Validate one positive LAMMPS atom-type ID.

    :param value: Value to validate.
    :return: A positive Python integer.
    :raises LammpsDataError: If the value is Boolean, nonintegral, or nonpositive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise LammpsDataError("atom type IDs must be integers")

    normalized = int(value)
    if normalized <= 0:
        raise LammpsDataError("atom type IDs must be positive")
    return normalized


def _normalize_type_mapping(
    type_dict: Mapping[object, object] | None,
) -> dict[int, str]:
    """Normalize a type map to ``type ID -> element symbol``.

    :param type_dict: Mapping in either ``str -> int`` or ``int -> str`` form.
    :return: A validated type-ID-to-species mapping.
    :raises LammpsDataError: If the mapping is malformed or ambiguous.
    """
    if type_dict is None:
        return {}
    if not isinstance(type_dict, Mapping):
        raise LammpsDataError(
            "type_dict must be a mapping[str, int] or mapping[int, str]"
        )
    if not type_dict:
        return {}

    items = list(type_dict.items())

    id_to_name_form = all(
        isinstance(key, Integral)
        and not isinstance(key, (bool, np.bool_))
        and isinstance(value, str)
        for key, value in items
    )
    name_to_id_form = all(
        isinstance(key, str)
        and isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        for key, value in items
    )

    if id_to_name_form:
        return {
            _strict_type_id(type_id): _strict_species_name(species)
            for type_id, species in items
        }

    if name_to_id_form:
        normalized: dict[int, str] = {}
        for species_value, type_id_value in items:
            species = _strict_species_name(species_value)
            type_id = _strict_type_id(type_id_value)

            previous = normalized.get(type_id)
            if previous is not None and previous != species:
                raise LammpsDataError(
                    f"atom type ID {type_id} is mapped to both "
                    f"{previous!r} and {species!r}"
                )
            normalized[type_id] = species

        return normalized

    raise LammpsDataError(
        "type_dict must be a mapping[str, int] or mapping[int, str]"
    )
