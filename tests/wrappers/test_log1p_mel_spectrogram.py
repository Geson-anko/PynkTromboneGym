import numpy as np

from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.spectrogram import calc_rfft_channel_num
from pynktrombonegym.wrappers import log1p_mel_spectrogram as l1pms

from ..test_env import target_sound_files


def test__init__():
    dflt = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files))

    assert dflt.n_mels == 80
    assert isinstance(dflt.mel_filter_bank, np.ndarray)
    assert dflt.mel_filter_bank.dtype == np.float32
    assert dflt.mel_filter_bank.shape == (dflt.n_mels, calc_rfft_channel_num(dflt.env.stft_window_size))

    f128 = l1pms.Log1pMelSpectrogram(PynkTrombone(target_sound_files), n_mels=128, dtype=np.float64)
    assert f128.mel_filter_bank.dtype == np.float64
    assert f128.mel_filter_bank.shape[0] == 128
