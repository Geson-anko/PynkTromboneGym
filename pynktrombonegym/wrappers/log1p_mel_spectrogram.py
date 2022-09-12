from typing import Any

import gym
import librosa
import numpy as np

from ..env import PynkTrombone


class Log1pMelSpectrogram(gym.ObservationWrapper):
    """This wrapper converts `target_sound_spectrogram` and
    `generated_sound_spectrogram` to log1p mel spectrogram.
    """

    env: PynkTrombone

    def __init__(self, env: PynkTrombone, n_mels: int = 80, new_step_api: bool = False, *, dtype: Any = np.float32):
        """Construct this wrapper

        Args:
            env (PynkTrombone): Base environment.
            n_mels (int): Channels of mel spectrogram.
            new_step_api (bool): See OpenAI gym docs.
        """
        super().__init__(env, new_step_api)
        self.n_mels = n_mels
        self.mel_filter_bank: np.ndarray = librosa.filters.mel(
            sr=env.sample_rate, n_fft=env.stft_window_size, n_mels=n_mels, dtype=dtype
        )
