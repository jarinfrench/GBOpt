# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define and validate physical topology along the boundary-normal axis.

This module owns the immutable topology vocabulary used to distinguish periodic
bicrystals, single-interface slabs, and structures whose boundary-normal topology is
unknown. It accepts enum members, serialized string values, or absent topology metadata
and returns a normalized :class:`BoundaryNormalTopology`.

Coordinate-based inference, external-file parsing, grain construction, interface
manipulation, and optimizer policy do not belong in this module.
"""

from __future__ import annotations

from enum import Enum


class BoundaryTopologyError(ValueError):
    """Raised when boundary-normal topology metadata is invalid."""


class BoundaryNormalTopology(str, Enum):
    """Explicit physical topology along the grain-boundary normal.

    ``PERIODIC_BICRYSTAL``
        The central grain boundary is accompanied by a second physical grain boundary
        across the periodic outer x faces.

    ``SINGLE_INTERFACE_SLAB``
        The structure contains one central grain boundary and free-surface or vacuum
        intervals that prevent the outer x faces from representing a second grain
        boundary.

    ``UNKNOWN``
        Available metadata is insufficient to establish either physical topology. This
        value represents missing information, not a known single-interface slab.
    """

    PERIODIC_BICRYSTAL = "periodic_bicrystal"
    SINGLE_INTERFACE_SLAB = "single_interface_slab"
    UNKNOWN = "unknown"


def normalize_boundary_normal_topology(
    value: BoundaryNormalTopology | str | None,
) -> BoundaryNormalTopology:
    """Return validated boundary-normal topology metadata.

    :param value: Topology as an enum member, its serialized string value, or ``None``
        when topology metadata is unavailable.
    :return: Validated boundary-normal topology. ``None`` normalizes to ``UNKNOWN``.
    :raises BoundaryTopologyError: If ``value`` is not a supported topology value.
    """
    if value is None:
        return BoundaryNormalTopology.UNKNOWN

    try:
        return BoundaryNormalTopology(value)
    except (TypeError, ValueError) as exc:
        raise BoundaryTopologyError(
            f"Unsupported boundary-normal topology: {value!r}"
        ) from exc
