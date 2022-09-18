from collections import OrderedDict
from typing import Any, Mapping, Optional

import gym
import numpy as np
from gym import spaces

from ..spaces import ActionSpaceNames as ASN


class ActionByAcceleration(gym.ActionWrapper):
    """Action Wrapper of PynkTrombone.
    Acording to physical system, the action will specify acceleration
    instead of position. By integrating it, it is converted to
    base environment action (position).

    It is expected that the initial random policy will be continuous and natural,
    and learning will be more easier.
    """

    velocities: Mapping
    positions: Mapping
    position_space: spaces.Space

    def __init__(
        self,
        env: gym.Env,
        action_scaler: float,
        initial_pos: Optional[Mapping] = None,
        new_step_api: bool = False,
    ) -> None:
        """Constuct this wrapper.

        Args:
            env (gym.Env): Base environment.
            action_scaler (float):  Scaling action with this value.
                It is recommended that `generate_chunk/sample_rate`
                be used because physical reason.
            initial_pos (Optinal[OrderedDict[str, np.ndarray]]): Initial position (action) of
                base environment. If None, this value is sampled randomly.
            new_step_api (bool): See OpenAI Gym API.
        """

        super().__init__(env, new_step_api)
        self.action_scaler = action_scaler
        self.position_space = env.action_space
        self.initial_pos = initial_pos

    @staticmethod
    def convert_space_to_acceleration(box_space: spaces.Box) -> spaces.Box:
        """Convert base action space to acceleration action space.
        Modify input `gym.spaces.Box` space so that one-half of
        the value range is 0.
        """
        rng = (box_space.high - box_space.low) / 2
        is_nan = np.isnan(rng)
        is_equal = box_space.high == box_space.low
        if np.nan in is_nan:
            rng[is_nan] = np.inf
            rng[np.logical_and(is_nan, is_equal)] == 0

        space = spaces.Box(-rng, rng, box_space.shape, box_space.dtype, box_space._np_random)  # type: ignore
        return space
