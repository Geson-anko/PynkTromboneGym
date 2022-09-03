from collections import OrderedDict
from typing import Sequence

import gym
import numpy as np
from gym import spaces
from pynktrombone import Voc

from . import spectrogram as spct
from .spaces import ActionSpace, ObservationSpace


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

        self.initialize_state()

        self.define_action_space()
        self.define_observation_space()
        self.define_reward_range()

    @property
    def target_sound_wave(self) -> np.ndarray:
        """Returns sliced `target_sound_wave_full` at `current_step`"""
        wave = self.target_sound_wave_full[
            self.current_step * self.generate_chunk : (self.current_step + 1) * self.generate_chunk
        ]
        return wave

    @property
    def generated_sound_wave(self) -> np.ndarray:
        """Returns generated sound wave at `current_step`"""
        return self._generated_sound_wave_2chunks[-self.generate_chunk :]

    def set_target_sound_files(self, file_paths: Sequence[str]) -> None:
        """Set `file_paths` to `self.target_sound_files`
        Args:
            file_paths (Iterable[str]): Paths to target sound files.
        """
        self.target_sound_files = file_paths

    def initialize_state(self) -> None:
        """Initialize this enviroment state."""
        self.current_step = 0
        self.target_sound_wave_full = self.load_sound_wave_randomly()
        self._generated_sound_wave_2chunks = np.zeros(self.generate_chunk * 2, dtype=np.float32)
        self.voc = Voc(self.sample_rate, self.generate_chunk, default_freq=self.default_frequency)

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
            ActionSpace(
                spaces.Box(-1.0, 1.0),
                spaces.Box(0.0, 1.0),
                spaces.Box(0, 3.5),
                spaces.Box(0, 3.5),
                spaces.Box(0, 3.5),
                spaces.Box(12, 40, dtype=int),
                spaces.Box(0, 3.5),
                spaces.Box(0, 1.5),
            ).to_dict()
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
            ObservationSpace(
                spaces.Box(-1.0, 1.0, (self.generate_chunk,)),
                spaces.Box(-1.0, 1.0, (self.generate_chunk,)),
                spaces.Box(0, float("inf"), spct_shape),
                spaces.Box(0, float("inf"), spct_shape),
                spaces.Box(0, self.sample_rate // 2),
                spaces.Box(-1.0, 1.0),
                spaces.Box(0.0, 1.0),
                spaces.Box(0.0, 5.0, (self.voc.tract_size,)),
                spaces.Box(0.0, 5.0, (self.voc.nose_size,)),
            ).to_dict()
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

    def get_target_sound_spectrogram(self) -> np.ndarray:
        """Slice target sound full wave and convert it to spectrogram.

        Returns:
            spectrogram (ndarray): Sliced target sound wave spectrogram.
                Shape -> (C, L)
                Dtype -> float32
        """
        if self.current_step == 0:
            wave = self.target_sound_wave_full[: self.generate_chunk]
        else:
            wave = self.target_sound_wave_full[
                (self.current_step - 1) * self.generate_chunk : (self.current_step + 1) * self.generate_chunk
            ]

        length = spct.calc_target_sound_spectrogram_length(
            self.generate_chunk, self.stft_window_size, self.stft_hop_length
        )
        spect = spct.stft(wave, self.stft_window_size, self.stft_hop_length)[-length:]
        spect = np.abs(spect).T.astype(np.float32)
        return spect

    def get_generated_sound_spectrogram(self) -> np.ndarray:
        """Convert generated sound wave to spectrogram

        There is `_generated_sound_wave_2chunks` as private variable,
        it contains previous and current generated wave for computing
        stft naturally.

        Returns:
            spectrogram (ndarray): A spectrogram of generated sound wave.
                Shape -> (C, L)
                Dtype -> float32
        """
        length = spct.calc_target_sound_spectrogram_length(
            self.generate_chunk, self.stft_window_size, self.stft_hop_length
        )

        spect = spct.stft(self._generated_sound_wave_2chunks, self.stft_window_size, self.stft_hop_length)
        spect = np.abs(spect[-length:]).T.astype(np.float32)
        return spect

    def get_current_observation(self) -> OrderedDict:
        """Return current observation.

        Return:
            observation (OrdereDict): observation.
        """
        target_sound_wave = self.target_sound_wave
        generated_sound_wave = self.generated_sound_wave
        target_sound_spectrogram = self.get_target_sound_spectrogram()
        generated_sound_spectrogram = self.get_generated_sound_spectrogram()
        frequency = self.voc.frequency
        pitch_shift = np.log2(frequency / self.default_frequency)
        tenseness = self.voc.tenseness
        tract_diameters = self.voc.current_tract_diameters
        nose_diameters = self.voc.nose_diameters

        obs = ObservationSpace(
            target_sound_wave,
            generated_sound_wave,
            target_sound_spectrogram,
            generated_sound_spectrogram,
            frequency,
            pitch_shift,
            tenseness,
            tract_diameters,
            nose_diameters,
        ).to_dict()

        return OrderedDict(obs)
