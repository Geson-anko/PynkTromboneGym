from typing import Sequence

import gym
import numpy as np
from gym import spaces
from pynktrombone import Voc

from . import spectrogram as spct


class PynkTrombone(gym.Env):
    r"""The vocal tract environment for speech generation.

    The main API methods that users of this class need to know are:

    - :meth:`__init__`  - Constructor of this enviroment.


    And set the following attributes:

    """

    def __init__(
        self,
        target_sound_files: Sequence[str],
        sample_rate: int = 44100,
        default_frequency: float = 400.0,
        generate_chunk: int = 512,
        stft_window_size: int = 1024,
        stft_hop_length: int = None,
    ):
        """Contructs environment. Setup `Voc`, deine spaces, and reset environment.

        Args:
            target_sound_files (Sequence[str]): Target sounds to imitate by vocal tract model.
            sample_rate (int): Resolution of sound wave.
                Target sounds and generation wave frequency are set to this.
            default_frequency (float): Base of glottis frequency.
            generate_chunk (int): Length generated in 1 step.
            stft_window_size (int): Window size of stft.
            stft_hop_length (int): Hop length of stft.
        """

        self.target_sound_files = target_sound_files
        self.sample_rate = sample_rate
        self.default_frequency = default_frequency
        self.generate_chunk = generate_chunk
        self.stft_window_size = stft_window_size
        if stft_hop_length is None:
            stft_hop_length = int(stft_window_size / 4)
        self.stft_hop_length = stft_hop_length

        self.voc = Voc(sample_rate, generate_chunk, default_freq=default_frequency)

        self.define_action_space()
        self.define_observation_space()
        self.define_reward_range()

    def set_target_sound_files(self, file_paths: Sequence[str]) -> None:
        """Set `file_paths` to `self.target_sound_files`
        Args:
            file_paths (Iterable[str]): Paths to target sound files.
        """
        self.target_sound_files = file_paths

    action_space: spaces.Dict

    def define_action_space(self) -> None:
        """Defines action space of this environment.

        Action space:
            pitch_shift,
            tenseness,
            traches,
            epiglottis,
            velum,
            tongue_index,
            tongue_diameter,
            lips
        """
        self.action_space = spaces.Dict(
            {
                "pitch_shift": spaces.Box(-1.0, 1.0),
                "tenseness": spaces.Box(0.0, 1.0),
                "trachea": spaces.Box(0, 3.5),
                "epiglottis": spaces.Box(0, 3.5),
                "velum": spaces.Box(0, 3.5),
                "tongue_index": spaces.Box(12, 40, dtype=int),
                "tongue_diameter": spaces.Box(0, 3.5),
                "lips": spaces.Box(0, 1.5),
            }
        )

    observation_space: spaces.Dict

    def define_observation_space(self) -> None:
        """Defines observation space of this enviroment.

        Observation space:
            target_sound,
            previous_generated_sound,
            current_frequency,
            current_pitch_shift,
            tenseness,
            current_tract_diameters,
            nose_diameters,
        """

        spct_shape = (
            spct.calc_rfft_channel_num(self.stft_window_size),
            spct.calc_target_sound_spectrogram_length(self.generate_chunk, self.stft_window_size, self.stft_hop_length),
        )

        self.observation_space = spaces.Dict(
            {
                "target_sound": spaces.Box(0, float("inf"), spct_shape),
                "previous_generated_sound": spaces.Box(0, float("inf"), spct_shape),
                "current_frequency": spaces.Box(0, self.sample_rate // 2),
                "current_pitch_shift": spaces.Box(-1.0, 1.0),
                "tenseness": spaces.Box(0.0, 1.0),
                "current_tract_diameters": spaces.Box(0.0, 5.0, (self.voc.tract_size,)),
                "nose_diameters": spaces.Box(0.0, 5.0, (self.voc.nose_size,)),
            }
        )

    def define_reward_range(self) -> None:
        """Define reward range of this environment.
        Reward is computed by measuring MSE between
        target_sound and generated_sound, and times -1.

        Range: [-inf, 0]
        """
        self.reward_range = (-float("inf"), 0.0)

    def load_sound_wave_randomly(self) -> np.ndarray:
        """Load sound file randomly.

        Return:
            waveform (ndarray): 1d numpy array, dtype is float32,
        """

        file_index = np.random.randint(0, len(self.target_sound_files))
        wave = spct.load_sound_file(self.target_sound_files[file_index], self.sample_rate)
        return wave
