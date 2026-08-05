# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pytest

from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    BoundaryTopologyError,
    normalize_boundary_normal_topology,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            None,
            BoundaryNormalTopology.UNKNOWN,
            id="missing-metadata",
        ),
        pytest.param(
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            id="periodic-enum",
        ),
        pytest.param(
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            id="slab-enum",
        ),
        pytest.param(
            BoundaryNormalTopology.UNKNOWN,
            BoundaryNormalTopology.UNKNOWN,
            id="unknown-enum",
        ),
        pytest.param(
            "periodic_bicrystal",
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            id="periodic-string",
        ),
        pytest.param(
            "single_interface_slab",
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            id="slab-string",
        ),
        pytest.param(
            "unknown",
            BoundaryNormalTopology.UNKNOWN,
            id="unknown-string",
        ),
    ],
)
def test_normalize_boundary_normal_topology(value, expected):
    assert normalize_boundary_normal_topology(value) is expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("periodic", id="unsupported-string"),
        pytest.param("", id="empty-string"),
        pytest.param(0, id="integer-zero"),
        pytest.param(1, id="integer-one"),
        pytest.param(True, id="boolean-true"),
        pytest.param(False, id="boolean-false"),
        pytest.param([], id="list"),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_normalize_rejects_unsupported_topology_values(value):
    with pytest.raises(
        BoundaryTopologyError,
        match="Unsupported boundary-normal topology",
    ):
        normalize_boundary_normal_topology(value)


def test_normalize_chains_original_conversion_error():
    with pytest.raises(BoundaryTopologyError) as exc_info:
        normalize_boundary_normal_topology("periodic")

    assert isinstance(exc_info.value.__cause__, ValueError)
