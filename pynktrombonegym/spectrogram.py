"""Tools for spectrogram operatings."""
import math
from typing import Any, Literal

import numpy as np
from pydub import AudioSegment

padT = Literal[
    "constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"
]


def calc_rfft_channel_num(window_size: int) -> int:
    """calculate output channel number of rfft operation at `window_size`.

    Args:
        window_size (int): wave length of fft.

    Returns:
        rfft_channel_num (int): Channel length of rfft output.
    """
    return int(window_size / 2) + 1


def calc_target_sound_spectrogram_length(chunk: int, window_size: int, hop_len: int) -> int:
    """Calculate output length of target sound spectrogram.

    Args:
        chunk (int): Generation length in one step.
        window_size (int): wave length of fft.
        hop_len (int): Step length of stft.

    Returns:
        calc_target_sound_spectrogram_length (int): Output time steps of stft.
    """

    return math.ceil(chunk / hop_len) + 1


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


def load_sound_file(file_path: Any, sample_rate: int) -> np.ndarray:
    """Load sound wave from file.
    Fix sound channel to 1, sample_rate, and
    normalize range to [-1,1].

    Args:
        file_path (str): Path to the sound file.

    Returns:
        waveform (ndarray): Range[-1,1]
            Waveform of loaded sound.
    """

    sound: AudioSegment = AudioSegment.from_file(file_path)

    if sound.frame_rate != sample_rate:
        sound = sound.set_frame_rate(sample_rate)
    if sound.channels != 1:
        sound = sound.set_channels(1)

    max_value = 2 ** (8 * sound.sample_width)
    wave = np.array(sound.get_array_of_samples()).reshape(-1) / max_value
    wave = wave.astype(np.float32)
    return wave
