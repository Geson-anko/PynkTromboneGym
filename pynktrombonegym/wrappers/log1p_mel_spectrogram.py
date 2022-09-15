from typing import Any

import gym
import librosa
import numpy as np
from gym import spaces

from ..env import PynkTrombone
from ..spaces import ObservationSpace
from ..spectrogram import calc_target_sound_spectrogram_length


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

        self.define_observation_space()

    observation_space: spaces.Dict

    def define_observation_space(self) -> None:
        """Re-defines observation space of this environment wrapper.
        This wrapper converts `target_sound_spectrogram` and
        `generated_sound_spectrogram` to log1p mel spectrogram.
        """
        obss = ObservationSpace.from_dict(self.env.observation_space)
        shape = (
            self.n_mels,
            calc_target_sound_spectrogram_length(
                self.env.generate_chunk, self.env.stft_window_size, self.env.stft_hop_length
            ),
        )
        log1p_mel_space = spaces.Box(0.0, float("inf"), shape)
        obss.target_sound_spectrogram = log1p_mel_space
        obss.generated_sound_spectrogram = log1p_mel_space

        self.observation_space = spaces.Dict(obss.to_dict())

    def log1p_mel(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert to log 1p mel spectrogram.
        Args:
            spectrogram (np.ndarray): 2d numpy array. (Channel, TimeStep)

        Returns:
            log1p mel spectrogram (np.ndarray): log1p mel scaled spectrogram.
                (n_mels, TimeStep)
        """
        return np.log1p(np.matmul(self.mel_filter_bank, spectrogram))

    def observation(self, observation):
        """Wrapps observation."""
        obs = ObservationSpace.from_dict(observation)
        obs.target_sound_spectrogram = self.log1p_mel(obs.target_sound_spectrogram)
        obs.generated_sound_spectrogram = self.log1p_mel(obs.generated_sound_spectrogram)
        return obs.to_dict()
