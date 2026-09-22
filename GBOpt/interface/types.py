# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the interface-candidate exception hierarchy.

This module holds only the exceptions raised by interface-domain validation. Atom
storage, grain-bounds validation, and defensive copying belong in ``model.py``, which is
the only other module in this package permitted to import from here.
"""

from __future__ import annotations


class InterfaceCandidateError(Exception):
    """Base class for exceptions raised by the interface-domain package."""


class InterfaceCandidateValueError(InterfaceCandidateError, ValueError):
    """Raised when interface-candidate state is malformed or internally inconsistent."""


class InterfaceCandidateTypeError(InterfaceCandidateError, TypeError):
    """Raised when an interface-candidate argument has an unsupported type."""


__all__ = [
    "InterfaceCandidateError",
    "InterfaceCandidateValueError",
    "InterfaceCandidateTypeError",
]
