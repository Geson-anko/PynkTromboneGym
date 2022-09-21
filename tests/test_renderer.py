from pynktrombonegym import renderer
from pynktrombonegym.env import PynkTrombone

from .test_env import target_sound_files


def test__init__():
    dflt = PynkTrombone(target_sound_files)
    rndr = renderer.Renderer(dflt)

    assert rndr.env is dflt
    assert rndr.figsize == (6.4, 4.8)

    rndr = renderer.Renderer(dflt, figsize=(1, 1))
    assert rndr.env is dflt
    assert rndr.figsize == (1, 1)
