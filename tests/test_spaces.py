from dataclasses import dataclass

import numpy as np

from pynktrombonegym import spaces as ptspaces


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


def test_ObservationSpace():
    cls = ptspaces.ObservationSpace

    obs = cls(
        np.zeros(10),
        np.ones(10),
        400.0,
        0.0,
        0.5,
        np.arange(10, dtype=float),
        np.arange(5, dtype=float),
    )

    assert np.all(obs.target_sound == np.zeros(10))
    assert np.all(obs.previous_generated_sound == np.ones(10))
    assert obs.current_frequency == 400.0
    assert obs.current_pitch_shift == 0.0
    assert obs.tenseness == 0.5
    assert np.all(obs.current_tract_diameters == np.arange(10, dtype=float))
    assert np.all(obs.nose_diameters == np.arange(5, dtype=float))
