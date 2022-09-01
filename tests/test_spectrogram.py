import numpy as np

from pynktrombonegym import spectrogram as spct

from .test_env import target_sound_files


def test_calc_rfft_channel_num():
    f = spct.calc_rfft_channel_num
    assert f(1024) == 513
    assert f(256) == 129
    assert f(400) == 201
    assert f(501) == 251


def test_calc_target_sound_spectrogram_length():
    f = spct.calc_target_sound_spectrogram_length
    assert f(512, 1024, 256) == 3
    assert f(1024, 1024, 256) == 5
    assert f(1024, 512, 128) == 9


def test_stft():
    f = spct.stft
    cf = spct.calc_rfft_channel_num
    lf = spct.calc_target_sound_spectrogram_length
    wave = np.sin(np.linspace(0, 2 * np.pi, 1024))
    out = f(wave, 1024, 256)
    assert out.dtype == np.complex128
    assert out.shape == (lf(len(wave), 1024, 256), cf(1024))


def test_load_sound_file():
    # missing sample rate assertion.
    f = spct.load_sound_file

    s1 = target_sound_files[0]
    w1 = f(s1, 44100)
    assert len(w1.shape) == 1
    assert np.abs(w1).max() <= 1
    assert w1.dtype == np.float32

    s2 = target_sound_files[1]
    w2 = f(s2, 22050)
    assert len(w2.shape) == 1
    assert np.abs(w2).max() <= 1
    assert w2.dtype == np.float32
