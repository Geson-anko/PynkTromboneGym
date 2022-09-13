import os

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.spaces import ObservationSpace
from pynktrombonegym.spectrogram import calc_rfft_channel_num, calc_target_sound_spectrogram_length
from pynktrombonegym.wrappers import log1p_mel_spectrogram as l1pms

from ..test_env import assert_space, target_sound_files

output_dir = "data/test_results"


def test__init__():
    dflt = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files))

    assert dflt.n_mels == 80
    assert isinstance(dflt.mel_filter_bank, np.ndarray)
    assert dflt.mel_filter_bank.dtype == np.float32
    assert dflt.mel_filter_bank.shape == (dflt.n_mels, calc_rfft_channel_num(dflt.env.stft_window_size))

    f128 = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files), n_mels=128, dtype=np.float64)
    assert f128.mel_filter_bank.dtype == np.float64
    assert f128.mel_filter_bank.shape[0] == 128


def test_define_observation_space():
    dflt = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files))
    # dflt.define_observation_space() # called in __init__
    obss = ObservationSpace.from_dict(dflt.observation_space)

    shape = (
        dflt.n_mels,
        calc_target_sound_spectrogram_length(
            dflt.env.generate_chunk, dflt.env.stft_window_size, dflt.env.stft_hop_length
        ),
    )
    space = spaces.Box(0.0, float("inf"), shape)
    assert_space(obss.target_sound_spectrogram, space)
    assert_space(obss.generated_sound_spectrogram, space)

    pure_obs = ObservationSpace.from_dict(dflt.env.observation_space)
    assert_space(obss.target_sound_wave, pure_obs.target_sound_wave)
    assert_space(obss.generated_sound_wave, pure_obs.generated_sound_wave)
    assert_space(obss.frequency, pure_obs.frequency)
    assert_space(obss.pitch_shift, pure_obs.pitch_shift)
    assert_space(obss.tenseness, pure_obs.tenseness)
    assert_space(obss.current_tract_diameters, pure_obs.current_tract_diameters)
    assert_space(obss.nose_diameters, pure_obs.nose_diameters)


def test_log1p_mel():
    n_mels = 80
    dflt = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files), n_mels=n_mels)

    spect = dflt.env.get_target_sound_spectrogram()
    log1p_mel = dflt.log1p_mel(spect)
    assert log1p_mel.shape == (n_mels, spect.shape[-1])

    target = np.log1p(np.matmul(dflt.mel_filter_bank, spect))
    assert np.mean(np.abs(target - log1p_mel)) < 1e-6

    fig, axises = plt.subplots(1, 2)
    ax0, ax1 = axises
    ax0.set_title("spectrogram")
    xs1, ys1 = spect.shape
    ax0.imshow(spect, aspect=ys1 / xs1)
    xm1, ym1 = log1p_mel.shape
    ax1.set_title("log1p mel spectrogram")
    ax1.imshow(log1p_mel, aspect=ym1 / xm1)

    filename = os.path.join(output_dir, f"{__name__}.test_log1p_mel.png")
    fig.savefig(filename)
    plt.close()


def test_observation():
    dflt = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files))
    obs = dflt.env.get_current_observation()
    wrapped = dflt.observation(obs)
    wrapped_obs = ObservationSpace.from_dict(wrapped)

    shape = (
        dflt.n_mels,
        calc_target_sound_spectrogram_length(
            dflt.env.generate_chunk, dflt.env.stft_window_size, dflt.env.stft_hop_length
        ),
    )
    assert wrapped_obs.target_sound_spectrogram.shape == shape
    assert wrapped_obs.generated_sound_spectrogram.shape == shape
