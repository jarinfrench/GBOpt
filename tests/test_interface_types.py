# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from GBOpt.interface.types import (
    InterfaceCandidateError,
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)


def test_interface_candidate_value_error_is_catchable_as_value_error():
    exc = InterfaceCandidateValueError("bad value")

    assert isinstance(exc, InterfaceCandidateError)
    assert isinstance(exc, ValueError)
    assert not isinstance(exc, TypeError)


def test_interface_candidate_type_error_is_catchable_as_type_error():
    exc = InterfaceCandidateTypeError("bad type")

    assert isinstance(exc, InterfaceCandidateError)
    assert isinstance(exc, TypeError)
    assert not isinstance(exc, ValueError)
