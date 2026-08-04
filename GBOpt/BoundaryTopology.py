"""Boundary-normal physical topology for GBOpt bicrystals and slabs."""

from __future__ import annotations

from enum import Enum


class BoundaryNormalTopology(str, Enum):
    """Physical topology along the grain-boundary normal direction."""

    PERIODIC_BICRYSTAL = "periodic_bicrystal"
    SINGLE_INTERFACE_SLAB = "single_interface_slab"
    UNKNOWN = "unknown"

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Whether the periodic outer x face is a second physical GB interface."""
        return self is BoundaryNormalTopology.PERIODIC_BICRYSTAL


def normalize_boundary_normal_topology(
    value: BoundaryNormalTopology | str | None,
    *,
    periodic_outer_x_interface: bool | None = None,
) -> BoundaryNormalTopology:
    """Normalize explicit topology while preserving legacy Boolean compatibility.

    A legacy ``True`` unambiguously denotes a periodic bicrystal. A legacy
    ``False`` does not distinguish a known slab from missing metadata, so it
    remains ``UNKNOWN`` unless an explicit topology is supplied.
    """
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
            raise ValueError(f"unsupported boundary-normal topology: {value!r}") from exc

    if periodic_outer_x_interface is not None:
        legacy = bool(periodic_outer_x_interface)
        if legacy != topology.periodic_outer_x_interface:
            if not (
                topology is BoundaryNormalTopology.UNKNOWN
                and legacy is False
            ):
                raise ValueError(
                    "periodic_outer_x_interface conflicts with normal_topology"
                )
    return topology
