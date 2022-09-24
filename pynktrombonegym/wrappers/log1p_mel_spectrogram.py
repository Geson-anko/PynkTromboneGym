from typing import Any, Sequence

import librosa
import numpy as np
from gym import spaces

from ..env import PynkTrombone
from ..spaces import ObservationSpaceNames as OSN
from ..spectrogram import calc_target_sound_spectrogram_length


class Log1pMelSpectrogram(PynkTrombone):
    """This is a subclass (not environment Wrapper!) of PynkTrombone.

    Convert `target_sound_spectrogram` and `generated_sound_spectrogram`
    to log1p mel spectrogram.

    Reward is computed with MSE of "log1p mel spectrogram".

    This class wraps:
    - :meth:`__init__` - create mel filter bank
    - :meth:`define_observation_space` - Convert to mel spectrogram shape.
    - :meth:`get_generated_sound_spectrogram` - Wraps with :meth:`log1p_mel`.
    - :meth:`get_target_sound_spectrogram` - Wraps with :meth:`log1p_mel`.

    And new attrs are:
    - :attr:`n_mels` - The mel spectrogram channels.
    - :attr:`mel_filter_bank` - The mel scale filter array.
    """

    def __init__(
        self, target_sound_files: Sequence[str], *args, n_mels: int = 80, dtype: Any = np.float32, **kwds
    ) -> None:
        """Construct this environment.
        Args:
            target_sound_files (Sequence[str]): Paths to target sound files.
            *args, **kwds: Please see BaseClass `PynkTrombone` environment.
            n_mels (int): Channels of mel spectrogram.
        """
        self.n_mels = n_mels
        super().__init__(target_sound_files, *args, **kwds)

        self.mel_filter_bank: np.ndarray = librosa.filters.mel(
            sr=self.sample_rate, n_fft=self.stft_window_size, n_mels=n_mels, dtype=dtype
        )

    def define_observation_space(self) -> spaces.Dict:
        """Wrapps base observation space."""
        obss = super().define_observation_space()
        shape = (
            self.n_mels,
            calc_target_sound_spectrogram_length(self.generate_chunk, self.stft_window_size, self.stft_hop_length),
        )
        log1p_mel_space = spaces.Box(0.0, float("inf"), shape)
        obss[OSN.TARGET_SOUND_SPECTROGRAM] = log1p_mel_space
        obss[OSN.GENERATED_SOUND_SPECTROGRAM] = log1p_mel_space

        return obss

    def log1p_mel(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert to log 1p mel spectrogram.
        Args:
            spectrogram (np.ndarray): 2d numpy array. (Channel, TimeStep)

        Returns:
            log1p mel spectrogram (np.ndarray): log1p mel scaled spectrogram.
                (n_mels, TimeStep)
        """
        return np.log1p(np.matmul(self.mel_filter_bank, spectrogram))

    def get_generated_sound_spectrogram(self) -> np.ndarray:
        """Convert to log1p mel spectrogram"""
        spect = super().get_generated_sound_spectrogram()
        return self.log1p_mel(spect)

    def get_target_sound_spectrogram(self) -> np.ndarray:
        """Convert to log1p mel spectrogram"""
        spect = super().get_target_sound_spectrogram()
        return self.log1p_mel(spect)
