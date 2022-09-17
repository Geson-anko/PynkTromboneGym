"""Definitions of Observation and Action space."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping


class ObservationSpaceNames:
    """Key names of Observation space."""

    TARGET_SOUND_WAVE = "target_sound_wave"
    GENERATED_SOUND_WAVE = "generated_sound_wave"
    TARGET_SOUND_SPECTROGRAM = "target_sound_spectrogram"
    GENERATED_SOUND_SPECTROGRAM = "generated_sound_spectrogram"
    FREQUENCY = "frequency"
    PITCH_SHIFT = "pitch_shift"
    TENSENESS = "tenseness"
    CURRENT_TRACT_DIAMETERS = "current_tract_diameters"
    NOSE_DIAMETERS = "nose_diameters"


class ActionSpaceNames:
    PITCH_SHIFT = "pitch_shift"
    TENSENESS = "tenseness"
    TRACHEA = "trachea"
    EPIGLOTTIS = "epiglottis"
    VELUM = "velum"
    TONGUE_INDEX = "tongue_index"
    TONGUE_DIAMETER = "tongue_diameter"
    LIPS = "lips"


@dataclass
class BaseSpace:
    """The BaseSpace for defining Observation and Action space."""

    def __post_init__(self):
        warnings.warn("Dataclass space defition is not recommended and will be removed in the future.", FutureWarning)

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
