"""Tools for spectrogram operatings."""
import math
from typing import Literal

import numpy as np

padT = Literal[
    "constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"
]


def calc_rfft_channel_num(n_fft: int) -> int:
    """calculate output channel number of rfft operation at `n_fft`.

    Args:
        n_fft (int): wave length of fft.

    Returns:
        rfft_channel_num (int): Channel length of rfft output.
    """
    return int(n_fft / 2) + 1


def stft(wave: np.ndarray, window_size: int, hop_len: int, padding_mode: padT = "reflect") -> np.ndarray:
    """Compute short time fourier transform
    Args:
        wave (ndarray): 1d array of waveform.
        window_size (int): Window length of a fft.
        hop_len (int): Step length of stft.
        padding_mode (str): Mode when filling in both ends of a waveform.

    Returns:
        Z (ndarray): STFT of wave.
            shape: (Steps, Channels), dtype: np.complex128
    """
    half_window = int(window_size / 2)
    wave_len = len(wave)

    hanning: np.ndarray = np.hanning(window_size)
    padded_wave: np.ndarray = np.pad(wave, (half_window, window_size), mode=padding_mode)
    waves: list[np.ndarray] = []

    for i in range(math.ceil(wave_len / hop_len) + 1):
        p = i * hop_len
        waves.append(padded_wave[p : p + window_size])

    waves = np.stack(waves)
    waves = waves * np.expand_dims(hanning, 0)
    Z = np.fft.rfft(waves, axis=1)

    return Z
