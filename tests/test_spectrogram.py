import numpy as np
import soundfile

from pynktrombonegym import spectrogram as spct

from .test_env import target_sound_files

sound_file_1sec = "data/1sec.wav"
sound_file_1sec_float32 = "data/1sec_float32.wav"
sound_file_1sec_ones = "data/1sec_ones.wav"


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

    w1sec_44100 = f(sound_file_1sec, 44100)
    assert len(w1sec_44100) == 44100
    w1sec_22050 = f(sound_file_1sec, 22050)
    assert len(w1sec_22050) == 22050

    expected, sample_rate = soundfile.read(sound_file_1sec_float32)
    w1sec_float32 = f(sound_file_1sec_float32, sample_rate)
    # 各要素の誤差がatol以上の時AssertionError
    np.testing.assert_allclose(w1sec_float32, expected)

    sound, sample_rate = soundfile.read(file=sound_file_1sec_ones, dtype=np.int16)
    expected = sound / 2**15
    w1sec_ones = f(sound_file_1sec_ones, sample_rate)
    np.testing.assert_equal(w1sec_ones, expected)


def test_pad_tail():
    f = spct.pad_tail

    src = np.arange(10, dtype=float)
    pad = np.zeros(5, dtype=float)
    assert np.all(f(src, 15) == np.concatenate([src, pad]))

    src = np.random.randn(3)
    pad = np.ones(2, dtype=float)
    assert np.all(f(src, 5, constant_values=1.0) == np.concatenate([src, pad]))

    # Check value error
    try:
        src = np.arange(100)
        f(src, 1)
        raise AssertionError
    except ValueError:
        pass
