"""Definitions of Observation and Action space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class BaseSpace:
    """The BaseSpace for defining Observation and Action space."""

    def to_dict(self) -> dict:
        """Return class members as dict.

        Returns:
            member dictionary (dict): Dictionary of member values.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Mapping) -> BaseSpace:
        """Create space from dict like object.

        Returns:
            Defined space (BaseSpace): Defined space object.

        Raises:
            If there is a key not exist in memeber, raises TypeError.
        """
        return cls(**d)


@dataclass
class ObservationSpace(BaseSpace):
    """"""


@dataclass
class ActionSpace(BaseSpace):
    """"""
