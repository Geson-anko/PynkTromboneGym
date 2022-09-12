"""Generation Test of PynkTromboneGym"""
import glob
import math
import os
from typing import Callable, Mapping

import numpy as np
import soundfile

from pynktrombonegym import env

target_sound_files = glob.glob("data/sample_target_sounds/*.wav")
output_dir = "data/test_results/generated_sounds"
sound_seconds = 5.0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def generate_sound(environment: env.PynkTrombone, action_fn: Callable, file_name: str) -> None:
    """Generate sound wave with environment and action_fn

    Args:
        enviroment (env.PynkTrombone): Vocal tract environment.
        action_fn (Callable): Return action for generating waves with environment.
            This funcion must be able receive `PynkTrombone` environment and
            return action.
            Ex:
            >>> def update_fn(environment: env.PynkTrombone):
            ...     return action

        file_name (str): The file name of generated sound.

    Returns:
        wave (np.ndarray): Generated wave. 1d array.
    """

    roop_num = math.ceil(sound_seconds / (environment.generate_chunk / environment.sample_rate))
    generated_waves = []
    environment.reset()
    for _ in range(roop_num):
        if environment.done:
            environment.reset()

        action = action_fn(environment)
        obs, _, _, _ = environment.step(action)
        obss = env.ObservationSpace.from_dict(obs)
        generated_waves.append(obss.generated_sound_wave)

    generated_sound_wave = np.concatenate(generated_waves).astype(np.float32)

    path = os.path.join(output_dir, file_name)
    soundfile.write(path, generated_sound_wave, environment.sample_rate)


def test_do_nothing():
    dflt = env.PynkTrombone(target_sound_files)

    def action_fn(e: env.PynkTrombone) -> Mapping:
        act = env.ActionSpace(
            pitch_shift=np.array([0.0]),
            tenseness=np.array([0.0]),
            trachea=np.array([0.6]),
            epiglottis=np.array([1.1]),
            velum=np.array([0.01]),
            tongue_index=np.array([20]),
            tongue_diameter=np.array([2.0]),
            lips=np.array([1.5]),
        ).to_dict()
        return act

    generate_sound(dflt, action_fn, f"{__name__}.test_do_nothing.wav")


def test_randomly():
    dflt = env.PynkTrombone(target_sound_files)

    def action_fn(e: env.PynkTrombone) -> Mapping:
        return e.action_space.sample()

    generate_sound(dflt, action_fn, f"{__name__}.test_randomly.wav")