# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define and validate physical topology along the boundary-normal axis.

This module owns the immutable topology vocabulary used to distinguish periodic
bicrystals, single-interface slabs, and structures whose boundary-normal topology is
unknown. It accepts explicit enum or serialized string values and normalizes the legacy
outer-interface Boolean into that explicit vocabulary.

Coordinate-based inference, external-file parsing, grain construction, interface
manipulation, and optimizer policy do not belong in this module.
"""

from __future__ import annotations

from enum import Enum


class BoundaryTopologyError(ValueError):
    """Raised when boundary-normal topology metadata is invalid or inconsistent."""


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
        state must not be treated as a known slab merely because the legacy
        periodic-interface flag is false.
    """

    PERIODIC_BICRYSTAL = "periodic_bicrystal"
    SINGLE_INTERFACE_SLAB = "single_interface_slab"
    UNKNOWN = "unknown"

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Return the legacy Boolean view of the outer x-face topology.

        This property returns ``True`` only for ``PERIODIC_BICRYSTAL``. It returns
        ``False`` for both ``SINGLE_INTERFACE_SLAB`` and ``UNKNOWN`` and therefore must
        not be used when those states need to be distinguished. Code making a
        topology-dependent physical decision must inspect the enum value directly.

        :return: Whether this topology is explicitly a periodic bicrystal.
        """
        return self is BoundaryNormalTopology.PERIODIC_BICRYSTAL


def normalize_boundary_normal_topology(
    value: BoundaryNormalTopology | str | None,
    *,
    periodic_outer_x_interface: bool | None = None,
) -> BoundaryNormalTopology:
    """Return validated boundary-normal topology metadata.

    ``periodic_outer_x_interface`` is retained only as a compatibility input. ``True``
    unambiguously identifies a periodic bicrystal. ``False`` does not distinguish a
    known single-interface slab from absent topology metadata, so it produces
    ``UNKNOWN`` when no explicit topology is supplied.

    :param value: Explicit topology as an enum member, its serialized string value, or
        ``None`` when explicit topology metadata is unavailable.
    :param periodic_outer_x_interface: Legacy compatibility flag. This must be exactly a
        Python ``bool`` or ``None``. Keyword argument, optional, defaults to ``None``.
    :return: Validated explicit boundary-normal topology.
    :raises BoundaryTopologyError: If either input has an unsupported value or if the
        explicit topology conflicts with the legacy compatibility flag.
    """
    if (
        periodic_outer_x_interface is not None
        and not isinstance(periodic_outer_x_interface, bool)
    ):
        raise BoundaryTopologyError("periodic_outer_x_interface must be a bool or None")

    if value is None:
        topology = (
            BoundaryNormalTopology.PERIODIC_BICRYSTAL
            if periodic_outer_x_interface is True
            else BoundaryNormalTopology.UNKNOWN
        )
    else:
        try:
            topology = BoundaryNormalTopology(value)
        except (TypeError, ValueError) as exc:
            raise BoundaryTopologyError(
                f"Unsupported boundary-normal topology: {value!r}"
            ) from exc

    if periodic_outer_x_interface is not None:
        legacy_is_periodic = periodic_outer_x_interface
        if legacy_is_periodic != topology.periodic_outer_x_interface:
            legacy_false_with_unknown_topology = (
                topology is BoundaryNormalTopology.UNKNOWN
                and legacy_is_periodic is False
            )
            if not legacy_false_with_unknown_topology:
                raise BoundaryTopologyError(
                    "periodic_outer_x_interface conflicts with boundary-normal topology"
                )

    return topology
