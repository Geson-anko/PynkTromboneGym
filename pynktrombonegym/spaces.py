"""Definitions of Observation and Action space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


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
    def from_dict(cls, d: Mapping):
        """Create space from dict like object.

        Returns:
            Defined space (BaseSpace): Defined space object.

        Raises:
            If there is a key not exist in memeber, raises TypeError.
        """
        return cls(**d)


@dataclass
class ObservationSpace(BaseSpace):
    """Defining observation space."""

    target_sound_wave: np.ndarray | Any
    generated_sound_wave: np.ndarray | Any
    target_sound_spectrogram: np.ndarray | Any
    generated_sound_spectrogram: np.ndarray | Any
    frequency: np.ndarray | Any
    pitch_shift: np.ndarray | Any
    tenseness: np.ndarray | Any
    current_tract_diameters: np.ndarray | Any
    nose_diameters: np.ndarray | Any


@dataclass
class ActionSpace(BaseSpace):
    """Defining action space."""

    pitch_shift: np.ndarray | Any
    tenseness: np.ndarray | Any
    trachea: np.ndarray | Any
    epiglottis: np.ndarray | Any
    velum: np.ndarray | Any
    tongue_index: np.ndarray | Any
    tongue_diameter: np.ndarray | Any
    lips: np.ndarray | Any
