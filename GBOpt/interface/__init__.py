# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the neutral interface-domain API.

The package-level surface contains the immutable interface-candidate value type and its
exception hierarchy. Private validation and defensive-copying helpers remain available
from ``model`` for this package's own use but are not promoted as user-facing API.
"""

from .model import InterfaceCandidate
from .types import (
    InterfaceCandidateError,
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)

__all__ = [
    # Exceptions
    "InterfaceCandidateError",
    "InterfaceCandidateValueError",
    "InterfaceCandidateTypeError",
    # Domain value
    "InterfaceCandidate",
]
