import copy
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces


class ActionByAcceleration(gym.ActionWrapper):
    """Action Wrapper of PynkTrombone environment.
    Acording to physical system, the action will specify acceleration
    instead of position. By integrating it, it is converted to
    base environment action (position).

    It is expected that the initial random policy will be continuous and natural,
    and learning will be more easier.

    The following methods are available in this wrapper.
    - :meth:`__init__` - Contruct this wrapper.
    - :meth:`convert_space_to_acceleration` - Static method. Convert Box space range to acceleration range.
    - :meth:`reset` - Reset this wrapping environment. Initialize this wrapper internal state.
    - Gym wrapper API.

    The following attributes are available.
    - :attr:`velocities` - The velocity of actions. Integrated acceleration action.
    - :attr:`positions` - The position of actions. Intergraed velocity action.
    - :attr:`position_space` - Base environment action space.
    """

    velocities: Dict
    positions: Dict
    position_space: spaces.Dict

    def __init__(
        self,
        env: gym.Env,
        action_scaler: float,
        initial_pos: Optional[Dict] = None,
        ignore_actions: Optional[Iterable[str]] = None,
        new_step_api: bool = True,
    ) -> None:
        """Constuct this wrapper.

        Args:
            env (gym.Env): Base environment.
            action_scaler (float):  Scaling action with this value.
                It is recommended that `generate_chunk/sample_rate`
                be used because physical reason.
            initial_pos (Optinal[OrderedDict[str, np.ndarray]]): Initial position (action) of
                base environment. If None, this value is sampled randomly.
            ignore_actions (Optional[Iterable[str]]): The action names that do not convert
                to acceleration action.
            new_step_api (bool): See OpenAI Gym API.
        """

        super().__init__(env, new_step_api)
        self.action_scaler = action_scaler
        self.position_space = env.action_space  # type: ignore
        self.initial_pos = initial_pos

        if ignore_actions is None:
            ignore_actions = set()
        self.ignore_actions = set(ignore_actions)

        self.action_space = self.define_action_space()

        self.initialize_state()

    @staticmethod
    def convert_space_to_acceleration(box_space: spaces.Box) -> spaces.Box:
        """Convert base action space to acceleration action space.
        Shift input `gym.spaces.Box` space so that one-half of
        the value range is 0.

        Args:
            box_spaces (spaces.Box): Box space of action.

        Returns:
            space (spaces.Box): Converted Box space.
        """
        rng = (box_space.high - box_space.low) / 2
        is_nan = np.isnan(rng)
        is_equal = box_space.high == box_space.low
        if np.nan in is_nan:
            rng[is_nan] = np.inf
            rng[np.logical_and(is_nan, is_equal)] == 0

        space = spaces.Box(-rng, rng, box_space.shape, box_space.dtype, box_space._np_random)  # type: ignore
        return space

    def define_action_space(self) -> spaces.Dict:
        """Define action space of this wrapper
        If action space is `gym.spaces.Box`, convert it to acceleration space.

        Returns:
            action_space (spaces.Dict): Convertd action spaces.
        """
        d = dict()
        for (k, v) in self.position_space.items():
            if k not in self.ignore_actions:
                v = self.convert_space_to_acceleration(v)
            d[k] = v

        return spaces.Dict(d)

    def initialize_state(self) -> None:
        """Initialize this state.
        This method called at :meth:`__init__` and :meth:`reset`.

        :attr:`velocities` are initialized with 0.
        If :attr:`initial_pos` is provided, :attr:`positions` are initialize with it.
        Else, initialized with random value.
        """
        if self.initial_pos is None:
            initial_pos = self.position_space.sample()
        else:
            initial_pos = copy.deepcopy(self.initial_pos)

        vel = OrderedDict()
        for (pos_key, pos_item) in copy.deepcopy(initial_pos).items():
            pos_item[:] = 0.0
            vel[pos_key] = pos_item

        self.velocities = vel
        self.positions = initial_pos

    def reset(self, **kwargs) -> Union[Any, Tuple[Any, dict]]:
        """Initialize state"""
        self.initialize_state()
        return super().reset(**kwargs)

    def action(self, action: Dict) -> Dict:
        """Convert acceleration action to positions.
        Note: :attr:`velocities` and :attr:`positions` are modified in this method.

        Args:
            action (Dict[str, np.ndarray]): acceleration action.

        Returns:
            positions (Dict): position action.
        """

        out_action = copy.deepcopy(action)
        for k in action.keys():
            if k not in self.ignore_actions:
                act = action[k] * self.action_scaler
                vel = self.velocities[k]
                pos = self.positions[k]
                pos_space: spaces.Box = self.position_space[k]  # type: ignore

                vel = vel + act
                pos = pos + vel
                is_limit = np.logical_or(pos < pos_space.low, pos_space.high < pos)
                vel[is_limit] = 0
                pos = np.clip(pos, pos_space.low, pos_space.high)

                self.velocities[k] = vel
                self.positions[k] = pos
                out_action[k] = pos

        return out_action
