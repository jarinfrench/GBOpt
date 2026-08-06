# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pytest

from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    BoundaryTopologyError,
    normalize_boundary_normal_topology,
)


@pytest.mark.parametrize(
    ("value", "legacy", "expected"),
    [
        pytest.param(
            None,
            None,
            BoundaryNormalTopology.UNKNOWN,
            id="missing-metadata",
        ),
        pytest.param(
            None,
            False,
            BoundaryNormalTopology.UNKNOWN,
            id="legacy-false-remains-unknown",
        ),
        pytest.param(
            None,
            True,
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            id="legacy-true-is-periodic",
        ),
        pytest.param(
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            None,
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            id="explicit-periodic-enum",
        ),
        pytest.param(
            "single_interface_slab",
            None,
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            id="explicit-slab-string",
        ),
        pytest.param(
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            False,
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            id="explicit-slab-compatible-with-false",
        ),
        pytest.param(
            BoundaryNormalTopology.UNKNOWN,
            False,
            BoundaryNormalTopology.UNKNOWN,
            id="explicit-unknown-compatible-with-false",
        ),
    ],
)
def test_normalize_boundary_normal_topology(value, legacy, expected):
    assert (
        normalize_boundary_normal_topology(
            value,
            periodic_outer_x_interface=legacy,
        )
        is expected
    )


@pytest.mark.parametrize(
    "legacy",
    [
        pytest.param(0, id="integer-zero"),
        pytest.param(1, id="integer-one"),
        pytest.param("", id="empty-string"),
        pytest.param("yes", id="truthy-string"),
        pytest.param([], id="empty-list"),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_normalize_rejects_non_boolean_legacy_values(legacy):
    with pytest.raises(
        BoundaryTopologyError,
        match="periodic_outer_x_interface must be a bool or None",
    ):
        normalize_boundary_normal_topology(
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            periodic_outer_x_interface=legacy,
        )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("periodic", id="unsupported-string"),
        pytest.param(1, id="integer"),
        pytest.param(True, id="boolean"),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_normalize_rejects_unsupported_explicit_topology(value):
    with pytest.raises(
        BoundaryTopologyError,
        match="Unsupported boundary-normal topology",
    ):
        normalize_boundary_normal_topology(value)


@pytest.mark.parametrize(
    ("topology", "legacy"),
    [
        pytest.param(
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            False,
            id="periodic-versus-false",
        ),
        pytest.param(
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            True,
            id="slab-versus-true",
        ),
        pytest.param(
            BoundaryNormalTopology.UNKNOWN,
            True,
            id="unknown-versus-true",
        ),
    ],
)
def test_normalize_rejects_conflicting_topology_metadata(topology, legacy):
    with pytest.raises(
        BoundaryTopologyError,
        match="conflicts with boundary-normal topology",
    ):
        normalize_boundary_normal_topology(
            topology,
            periodic_outer_x_interface=legacy,
        )


def test_legacy_boolean_property_is_explicitly_lossy():
    assert BoundaryNormalTopology.PERIODIC_BICRYSTAL.periodic_outer_x_interface is True
    assert (
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB.periodic_outer_x_interface is False
    )
    assert BoundaryNormalTopology.UNKNOWN.periodic_outer_x_interface is False
