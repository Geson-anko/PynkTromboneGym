import copy
import math
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from pynktrombone import Voc

from . import spectrogram as spct
from .spaces import ActionSpace, ObservationSpace

RenderFrame = TypeVar("RenderFrame", plt.Figure, np.ndarray)


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
        generate_chunk: int = 1024,
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

        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.define_reward_range()

    @property
    def target_sound_wave(self) -> np.ndarray:
        """Returns sliced `target_sound_wave_full` at `current_step`"""
        wave = self.target_sound_wave_full[
            self.current_step * self.generate_chunk : (self.current_step + 1) * self.generate_chunk
        ]
        return spct.pad_tail(wave, self.generate_chunk)

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
        self._stored_state_figures: List[plt.Figure] = []

    action_space: spaces.Dict

    def define_action_space(self) -> spaces.Dict:
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
        action_space = spaces.Dict(
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
        return action_space

    observation_space: spaces.Dict

    def define_observation_space(self) -> spaces.Dict:
        """Defines observation space of this enviroment.

        Observation space:
            target_sound_spectrogram,
            generated_sound_spectrogram,
            frequency,
            pitch_shift,
            tenseness,
            current_tract_diameters,
            nose_diameters,
        """

        spct_shape = (
            spct.calc_rfft_channel_num(self.stft_window_size),
            spct.calc_target_sound_spectrogram_length(self.generate_chunk, self.stft_window_size, self.stft_hop_length),
        )

        observation_space = spaces.Dict(
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

        return observation_space

    def define_reward_range(self) -> None:
        """Define reward range of this environment.
        Reward is computed by measuring MSE between
        target_sound_spectrogram and generated_sound, and times -1.

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
            wave = spct.pad_tail(wave, self.generate_chunk)
        else:
            wave = self.target_sound_wave_full[
                (self.current_step - 1) * self.generate_chunk : (self.current_step + 1) * self.generate_chunk
            ]
            wave = spct.pad_tail(wave, 2 * self.generate_chunk)

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
        frequency = np.array([self.voc.frequency], dtype=np.float32)
        pitch_shift = np.log2(frequency / self.default_frequency)
        tenseness = np.array([self.voc.tenseness], dtype=np.float32)
        tract_diameters = self.voc.current_tract_diameters.astype(np.float32)
        nose_diameters = self.voc.nose_diameters.astype(np.float32)

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

    def reset(
        self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None
    ) -> OrderedDict:
        """Reset this enviroment.
        Choice sound file randomly and load waveform at random start point.
        Internal vocal tract model `Voc` is reconstructed too.

        Returns initial `target_sound` spectrogram from loaded sound and
        same shape zero array as initial `previous_generated_sound`.

        Returns:
            observation (OrderedDict): Initial observation of this enviroment.
        """
        super().reset(seed=seed, return_info=return_info, options=options)

        self.initialize_state()
        obs = self.get_current_observation()
        return obs

    def compute_reward(self) -> float:
        """Compute current reward.
        Measure 'minus' MSE between target and generated.

        Returns:
            reward (float):  Computed reward value.
        """
        target = self.get_target_sound_spectrogram()
        generated = self.get_generated_sound_spectrogram()

        return -mean_squared_error(generated, target)

    @property
    def done(self) -> bool:
        """Check if enviroment has done."""
        return self.current_step * self.generate_chunk >= len(self.target_sound_wave_full)

    @property
    def max_steps(self) -> int:
        """Returns max step number of this environment."""
        return math.ceil(len(self.target_sound_wave_full) / self.generate_chunk)

    def step(self, action: Mapping) -> Tuple[OrderedDict, float, bool, dict]:
        """Step this enviroment by action.

        Args:
            action (OrderedDict): Dict of action values.

        Returns:
            observation (OrderedDict): Next step observation.
            reward (float): Reward of current step.
            done (bool): Whether the environment has been finished or not.
            info (dict): Debug informations.

        Raises:
            RuntimeError: If done is True, raises runtime error.
                Please call `reset` method of this enviroment.
        """
        if self.done:
            raise RuntimeError("This environment has been finished. Please call `reset` method.")

        info: Dict[Any, Any] = dict()

        acts = ActionSpace.from_dict(action)
        self.voc.frequency = self.default_frequency * (2**acts.pitch_shift)
        self.voc.tenseness = acts.tenseness
        self.voc.set_tract_parameters(
            acts.trachea.item(),
            acts.epiglottis.item(),
            acts.velum.item(),
            acts.tongue_index.item(),
            acts.tongue_diameter.item(),
            acts.lips.item(),
        )

        generated_wave = self.voc.play_chunk()
        self._generated_sound_wave_2chunks = np.concatenate([self.generated_sound_wave, generated_wave])
        reward = self.compute_reward()  # Minus error between generated and 'current' target.

        ##### Next step #####
        self.current_step += 1
        done = self.done
        obs = self.get_current_observation()
        return obs, reward, done, info

    def create_state_figure(self) -> plt.Figure:
        """Create a figure of current environment state.

        Plotting:
        - current_step
        - current_tract_diameters
        - nose_diameters
        - current voc frequency
        - current voc tenseness

        Returns:
            figure (plt.Figure): A figure of current environment state.
        """
        obs = ObservationSpace.from_dict(self.get_current_observation())

        fig = plt.figure(figsize=(6.4 * 1.5, 4.8 * 1.5))
        ax = fig.add_subplot(1, 1, 1)

        indices = list(range(self.voc.tract_size))
        nose_indices = indices[-self.voc.nose_size :]
        ax.set_ylim(0.0, 5.0)
        ax.plot(nose_indices, obs.nose_diameters, label="nose diameters")
        ax.plot(indices, obs.current_tract_diameters, label="tract diameters")
        ax.legend()

        ax.set_title("Tract diameters")
        ax.set_xlabel("diameter index")
        ax.set_ylabel("diameter [cm]")

        info = (
            f"current step: {self.current_step}\n"
            f"frequency: {obs.frequency.item(): .2f}\n"
            f"tenseness: {obs.tenseness.item(): .2f}\n"
        )

        ax.text(1, 4.0, info)

        return fig

    def render(
        self, mode: Optional[Literal["figures", "rgb_arrays", "single_figure", "single_rgb_array"]] = None
    ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Render a figure of current Vocal Tract diameters and etc.

        Args:
            mode (Optional[Literal]): Rendering mode.
                - None: Create figure and store it.
                - "figures": Render current figure and return all stored figures.
                - "rgb_arrays": Render current figure array and return all
                    stored figure array.
                - "single_rgb_array": Return RGB image array of figure.
                - "singe_figure": Return a figure of matplotlib.pyplot.
                Note: "figures" and "rgb_arrays" mode clear all stored figures.
                    If you want all figures and arrays, Use "figures" mode and
                    `fig2rgba_array` function to convert figures to numpy array.

        Returns:
            image (Optional[Union[RenderFrame, List[RenderFrame]]]): Renderd image or images.

        Raises:
            NotImplementedError: When mode is unexpected value.
        """

        fig = self.create_state_figure()

        if mode is None:
            self._stored_state_figures.append(fig)
            return None
        elif mode == "single_figure":
            return fig
        elif mode == "single_rgb_array":
            return fig2rgba_array(fig)[:, :, :3]

        figures = copy.copy(self._stored_state_figures)
        figures.append(fig)
        self._stored_state_figures.clear()

        if mode == "figures":
            return figures
        elif mode == "rgb_arrays":
            return [fig2rgba_array(f)[:, :, :3] for f in figures]
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented!")


def fig2rgba_array(figure: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array.

    Args:
        figure (plt.Figure): A matplotlib figure.

    Returns:
        image array (np.ndarray): Numpy array of rendered figure.
            Shape: (Height, Width, RGBA)
    """

    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(h * 2, w * 2, 4).copy()
    buf = np.roll(buf, 3, axis=-1)
    return buf


def mean_squared_error(output: np.ndarray, target: np.ndarray) -> float:
    """Compute mse.
    Output and Target must have same shape.

    Args:
        output (ndarray): The output of model.
        target (ndarray): Target of output

    Returns:
        mse (float): Mean Squared Error.
    """
    delta = output - target
    mse = float(np.sum(delta * delta / target.size))
    return mse
