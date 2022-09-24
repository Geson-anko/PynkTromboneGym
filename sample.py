import glob
import math
import os
from typing import Callable

import gym
import numpy as np
import soundfile

from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from pynktrombonegym.wrappers import ActionByAcceleration, Log1pMelSpectrogram

target_sound_files = glob.glob("data/sample_target_sounds/*.wav")
sound_seconds = 5.0
output_dir = "data/test_results"


def generate_sound(environment: gym.Env, action_fn: Callable, file_name: str, generate_chunk: int, sample_rate) -> None:
    """Generate sound wave with environment and action_fn

    Args:
        enviroment (env.PynkTrombone): Vocal tract environment.
        action_fn (Callable): Return action for generating waves with environment.
            This funcion must be able to receive `PynkTrombone` environment and
            to return action.
            Ex:
            >>> def update_fn(environment: gym.Env):
            ...     return action

        file_name (str): The file name of generated sound.

    Returns:
        wave (np.ndarray): Generated wave. 1d array.
    """

    roop_num = math.ceil(sound_seconds / (generate_chunk / sample_rate))
    generated_waves = []
    environment.reset()
    done = False
    for _ in range(roop_num):
        if done:
            environment.reset()

        action = action_fn(environment)
        obs, _, done, _ = environment.step(action)  # type: ignore
        generated_sound_wave = obs[OSN.GENERATED_SOUND_WAVE]
        generated_waves.append(generated_sound_wave)

    generated_sound_wave = np.concatenate(generated_waves).astype(np.float32)

    path = os.path.join(output_dir, file_name)
    soundfile.write(path, generated_sound_wave, sample_rate)


if __name__ == "__main__":
    env = Log1pMelSpectrogram(target_sound_files)
    action_scaler = env.generate_chunk / env.sample_rate
    wrapped = ActionByAcceleration(env, action_scaler)

    def action_fn(e: gym.Env) -> dict:
        return e.action_space.sample()

    generate_sound(wrapped, action_fn, "generated_randomly.wav", env.generate_chunk, env.sample_rate)

    wrapped.close()  # you must call.
