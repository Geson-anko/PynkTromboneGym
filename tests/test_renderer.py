import gc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines, text
from pynktrombone.voc import Voc

from pynktrombonegym import renderer


def test__init__():
    voc = Voc()
    rndr = renderer.Renderer(voc)

    assert rndr.voc is voc
    assert rndr.figsize == (6.4, 4.8)

    rndr = renderer.Renderer(voc, figsize=(1, 1))
    assert rndr.voc is voc
    assert rndr.figsize == (1, 1)


def test_make_infomation_text():
    voc = Voc()
    rndr = renderer.Renderer(voc)

    info_text = rndr.make_infomation_text()

    correct = f"frequency: {float(voc.frequency): .2f}\n" f"tenseness: {float(voc.tenseness): .2f}\n"

    assert info_text == correct


def test_create_initial_components():
    voc = Voc()
    rndr = renderer.Renderer(voc)
    # called in `__init__`

    assert isinstance(rndr.figure, plt.Figure)
    assert isinstance(rndr.axes, plt.Axes)
    assert rndr.indices == list(range(voc.tract_size))
    assert rndr.nose_indices == rndr.indices[-voc.nose_size :]
    assert isinstance(rndr.nose_diameters_line, lines.Line2D)
    assert isinstance(rndr.tract_diameters_line, lines.Line2D)
    assert isinstance(rndr.infomation_text, text.Text)

    rndr.figure.savefig(f"data/test_results/{__name__}.test_create_initial_components.png")


def test_update_values():
    voc = Voc()
    rndr = renderer.Renderer(voc)

    voc.frequency = voc.frequency + 10
    voc.tenseness = 0.44
    voc.set_tract_parameters(0.7, 0.5, 0.1, 29, 1.0, 1.0)
    voc.step()

    rndr.update_values()

    assert rndr.infomation_text.get_text() == rndr.make_infomation_text()

    rndr.figure.savefig(f"data/test_results/{__name__}.update_values.png")


def test_fig2rgba_array():
    voc = Voc()
    rndr = renderer.Renderer(voc)

    array = rndr.fig2rgba_array(rndr.figure)

    assert isinstance(array, np.ndarray)
    assert array.shape[-1] == 4
    assert array.dtype == np.uint8
    assert array.ndim == 3
    w, h = rndr.figure.canvas.get_width_height(physical=True)
    assert array.shape[:2] == (h, w)

    plt.imsave(f"data/test_results/{__name__}.test_fig2rgba_array.png", array)


def test_fig2rgb_array():
    voc = Voc()
    rndr = renderer.Renderer(voc)

    array = rndr.fig2rgb_array(rndr.figure)

    assert isinstance(array, np.ndarray)
    assert array.shape[-1] == 3
    plt.imsave(f"data/test_results/{__name__}.test_fig2rgb_array.png", array)


def test_render_rgb_array():
    voc = Voc()
    rndr = renderer.Renderer(voc)

    array = rndr.render_rgb_array()

    assert isinstance(array, np.ndarray)
    assert array.shape[-1] == 3
    assert array.ndim == 3
    assert array.dtype == np.uint8
    plt.imsave(f"data/test_results/{__name__}.test_render_rgb_array.png", array)


def test_close():
    gc.collect()
    voc = Voc()
    fignum_before_create = len(plt.get_fignums())
    rndr = renderer.Renderer(voc)
    fignum_after_create = len(plt.get_fignums())
    axnum = len(rndr.figure.axes)

    assert fignum_before_create + 1 == fignum_after_create
    assert axnum > 0

    rndr.close()

    assert fignum_before_create == len(plt.get_fignums())
    assert len(rndr.figure.axes) == 0


def test__del__():
    gc.collect()
    voc = Voc()
    fignum_before_create = len(plt.get_fignums())
    rndr = renderer.Renderer(voc)
    fignum_after_create = len(plt.get_fignums())

    assert fignum_before_create + 1 == fignum_after_create
    rndr.__del__()
    assert fignum_before_create == len(plt.get_fignums())
