import matplotlib.pyplot as plt
import numpy as np
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


def test_create_initial_components():
    dflt = PynkTrombone(target_sound_files)
    rndr = renderer.Renderer(dflt)
    # called in `__init__`

    assert isinstance(rndr.figure, plt.Figure)
    assert isinstance(rndr.axes, plt.Axes)
    assert rndr.indices == list(range(dflt.voc.tract_size))
    assert rndr.nose_indices == rndr.indices[-dflt.voc.nose_size :]
    assert isinstance(rndr.nose_diameters_line, lines.Line2D)
    assert isinstance(rndr.tract_diameters_line, lines.Line2D)
    assert isinstance(rndr.infomation_text, text.Text)

    rndr.figure.savefig(f"data/test_results/{__name__}.test_create_initial_components.png")


def test_update_values():
    dflt = PynkTrombone(target_sound_files)
    rndr = renderer.Renderer(dflt)

    dflt.reset()
    action = dflt.action_space.sample()
    dflt.step(action)

    rndr.update_values()

    assert rndr.infomation_text.get_text() == rndr.make_infomation_text()

    rndr.figure.savefig(f"data/test_results/{__name__}.update_values.png")


def test_fig2rgba_array():
    dflt = PynkTrombone(target_sound_files)
    rndr = renderer.Renderer(dflt)

    array = rndr.fig2rgba_array(rndr.figure)

    assert isinstance(array, np.ndarray)
    assert array.shape[-1] == 4
    assert array.dtype == np.uint8
    assert array.ndim == 3
    w, h = rndr.figure.canvas.get_width_height(physical=True)
    assert array.shape[:2] == (h, w)

    plt.imsave(f"data/test_results/{__name__}.test_fig2rgba_array.png", array)
