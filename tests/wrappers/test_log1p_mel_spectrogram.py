import os

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from pynktrombonegym.environment import PynkTrombone
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from pynktrombonegym.spectrogram import (
    calc_rfft_channel_num,
    calc_target_sound_spectrogram_length,
)
from pynktrombonegym.wrappers import log1p_mel_spectrogram as l1pms

from ..test_environment import assert_space, target_sound_files

output_dir = "data/test_results"


def test__init__():
    dflt = l1pms.Log1pMelSpectrogram(target_sound_files)

    assert dflt.n_mels == 80
    assert isinstance(dflt.mel_filter_bank, np.ndarray)
    assert dflt.mel_filter_bank.dtype == np.float32
    assert dflt.mel_filter_bank.shape == (dflt.n_mels, calc_rfft_channel_num(dflt.stft_window_size))

    f128 = l1pms.Log1pMelSpectrogram(target_sound_files, n_mels=128, dtype=np.float64)
    assert f128.mel_filter_bank.dtype == np.float64
    assert f128.mel_filter_bank.shape[0] == 128


def test_define_observation_space():
    dflt = l1pms.Log1pMelSpectrogram(target_sound_files)
    # dflt.define_observation_space() # called in __init__
    obs = dflt.define_observation_space()

    shape = (
        dflt.n_mels,
        calc_target_sound_spectrogram_length(dflt.generate_chunk, dflt.stft_window_size, dflt.stft_hop_length),
    )
    space = spaces.Box(0.0, float("inf"), shape)
    assert_space(obs[OSN.TARGET_SOUND_SPECTROGRAM], space)
    assert_space(obs[OSN.GENERATED_SOUND_SPECTROGRAM], space)
    sampled_obs = dflt.observation_space.sample()
    assert sampled_obs[OSN.GENERATED_SOUND_SPECTROGRAM].shape == space.shape
    assert sampled_obs[OSN.GENERATED_SOUND_SPECTROGRAM].dtype == space.dtype
    assert sampled_obs[OSN.TARGET_SOUND_SPECTROGRAM].shape == space.shape
    assert sampled_obs[OSN.TARGET_SOUND_SPECTROGRAM].dtype == space.dtype

    pure_obs = PynkTrombone(target_sound_files).observation_space
    assert_space(obs[OSN.TARGET_SOUND_WAVE], pure_obs[OSN.TARGET_SOUND_WAVE])
    assert_space(obs[OSN.GENERATED_SOUND_WAVE], pure_obs[OSN.GENERATED_SOUND_WAVE])
    assert_space(obs[OSN.FREQUENCY], pure_obs[OSN.FREQUENCY])
    assert_space(obs[OSN.PITCH_SHIFT], pure_obs[OSN.PITCH_SHIFT])
    assert_space(obs[OSN.TENSENESS], pure_obs[OSN.TENSENESS])
    assert_space(obs[OSN.CURRENT_TRACT_DIAMETERS], pure_obs[OSN.CURRENT_TRACT_DIAMETERS])
    assert_space(obs[OSN.NOSE_DIAMETERS], pure_obs[OSN.NOSE_DIAMETERS])


def test_log1p_mel():
    n_mels = 80
    dflt = l1pms.Log1pMelSpectrogram(target_sound_files[:1], n_mels=n_mels)

    base = PynkTrombone(target_sound_files)
    spect = base.get_target_sound_spectrogram()
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


def test_observation_wrapping():
    dflt = l1pms.Log1pMelSpectrogram(target_sound_files)
    wrapped_obs = dflt.get_current_observation()

    shape = (
        dflt.n_mels,
        calc_target_sound_spectrogram_length(dflt.generate_chunk, dflt.stft_window_size, dflt.stft_hop_length),
    )
    assert wrapped_obs[OSN.TARGET_SOUND_SPECTROGRAM].shape == shape
    assert wrapped_obs[OSN.GENERATED_SOUND_SPECTROGRAM].shape == shape
