from dataclasses import dataclass

import numpy as np

from pynktrombonegym import spaces as ptspaces


def assert_space_names(name_cls: object):
    for k in name_cls.__dict__.keys():
        if not k.startswith("__"):
            assert k.isupper(), "Key name must be upper case because it is constant value."
            assert k == getattr(name_cls, k).upper()


def test_ObservationSpaceNames():
    name_cls = ptspaces.ObservationSpaceNames
    assert_space_names(name_cls)

    assert name_cls.TARGET_SOUND_WAVE == "target_sound_wave"
    assert name_cls.GENERATED_SOUND_WAVE == "generated_sound_wave"
    assert name_cls.TARGET_SOUND_SPECTROGRAM == "target_sound_spectrogram"
    assert name_cls.GENERATED_SOUND_SPECTROGRAM == "generated_sound_spectrogram"
    assert name_cls.FREQUENCY == "frequency"
    assert name_cls.PITCH_SHIFT == "pitch_shift"
    assert name_cls.TENSENESS == "tenseness"
    assert name_cls.CURRENT_TRACT_DIAMETERS == "current_tract_diameters"
    assert name_cls.NOSE_DIAMETERS == "nose_diameters"


def test_ActionSpaceNames():
    name_cls = ptspaces.ActionSpaceNames
    assert_space_names(name_cls)

    assert name_cls.PITCH_SHIFT == "pitch_shift"
    assert name_cls.TENSENESS == "tenseness"
    assert name_cls.TRACHEA == "trachea"
    assert name_cls.EPIGLOTTIS == "epiglottis"
    assert name_cls.VELUM == "velum"
    assert name_cls.TONGUE_INDEX == "tongue_index"
    assert name_cls.TONGUE_DIAMETER == "tongue_diameter"
    assert name_cls.LIPS == "lips"


def test_BaseSpace():
    bs = ptspaces.BaseSpace()

    d = bs.to_dict()
    assert type(d) is dict
    assert d == dict()
    try:
        ptspaces.BaseSpace.from_dict({"a": 0})
        raise AssertionError
    except TypeError:
        pass

    @dataclass
    class Space(ptspaces.BaseSpace):
        state1: float
        state2: int
        state3: str

    o = Space(1.0, 2, "3")
    assert o.to_dict() == {"state1": 1.0, "state2": 2, "state3": "3"}

    d = {"state1": 0.0, "state2": 1, "state3": "2"}
    o = Space.from_dict(d)
    assert o.state1 == 0.0
    assert o.state2 == 1
    assert o.state3 == "2"
