# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the explicit manipulation-operation registry.

``ManipulationRegistry`` maps operation names to ``Manipulation`` implementations for
lookup by ``GBManipulator.apply_named()``. Registration is always explicit: importing
this module (or the package) registers nothing on its own, so no built-in or
third-party operation becomes available merely by being importable.
"""

from __future__ import annotations

from GBOpt.manipulation.types import (
    Manipulation,
    ManipulationLookupError,
    ManipulationRegistrationError,
)


class ManipulationRegistry:
    """An explicit, mutable name-to-operation lookup table."""

    def __init__(self) -> None:
        """Construct an empty registry."""
        self.__entries: dict[str, Manipulation] = {}

    def register(self, name: str, manipulation: Manipulation) -> None:
        """Register ``manipulation`` under ``name``.

        :param name: Name future callers will use to look up ``manipulation``.
        :param manipulation: Operation implementation to register.
        :raises ManipulationRegistrationError: If ``name`` is already registered.
        """
        if name in self.__entries:
            raise ManipulationRegistrationError(
                f"a manipulation named {name!r} is already registered"
            )
        self.__entries[name] = manipulation

    def get(self, name: str) -> Manipulation:
        """Return the operation registered under ``name``.

        :param name: Registered operation name.
        :return: The registered operation.
        :raises ManipulationLookupError: If ``name`` is not registered.
        """
        try:
            return self.__entries[name]
        except KeyError as exc:
            raise ManipulationLookupError(
                f"no manipulation named {name!r} is registered"
            ) from exc


default_registry = ManipulationRegistry()


__all__ = ["ManipulationRegistry", "default_registry"]
