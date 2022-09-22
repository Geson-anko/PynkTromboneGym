import matplotlib.pyplot as plt
from matplotlib import lines, text

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


def test_make_infomation_text():
    dflt = PynkTrombone(target_sound_files)
    rndr = renderer.Renderer(dflt)

    info_text = rndr.make_infomation_text()

    correct = (
        f"current step: {dflt.current_step}\n"
        f"frequency: {float(dflt.voc.frequency): .2f}\n"
        f"tenseness: {float(dflt.voc.tenseness): .2f}\n"
    )

    assert info_text == correct
