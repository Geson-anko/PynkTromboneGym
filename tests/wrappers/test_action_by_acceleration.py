import copy
import math
from collections import OrderedDict

import numpy as np
from gym import spaces

from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.spaces import ActionSpaceNames as ASN
from pynktrombonegym.wrappers.action_by_acceleration import ActionByAcceleration

from ..test_env import assert_dict_space_key, assert_space, target_sound_files

initial_pos = OrderedDict(
    {
        ASN.PITCH_SHIFT: np.array([0.0]),
        ASN.TENSENESS: np.array([0.0]),
        ASN.TRACHEA: np.array([0.6]),
        ASN.EPIGLOTTIS: np.array([1.1]),
        ASN.VELUM: np.array([0.01]),
        ASN.TONGUE_INDEX: np.array([20]),
        ASN.TONGUE_DIAMETER: np.array([2.0]),
        ASN.LIPS: np.array([1.5]),
    }
)
assert_dict_space_key(initial_pos, ASN)

example_action = OrderedDict(
    {
        ASN.PITCH_SHIFT: np.array([0.1]),
        ASN.TENSENESS: np.array([0.2]),
        ASN.TRACHEA: np.array([-0.3]),
        ASN.EPIGLOTTIS: np.array([-1.1]),
        ASN.VELUM: np.array([-0.1]),
        ASN.TONGUE_INDEX: np.array([-2.0]),
        ASN.TONGUE_DIAMETER: np.array([-1.0]),
        ASN.LIPS: np.array([0.2]),
    }
)


def test__init__():
    base = PynkTrombone(target_sound_files)

    action_scaler = 1.0
    wrapped = ActionByAcceleration(base, action_scaler)

    assert wrapped.action_scaler == action_scaler
    assert wrapped.position_space is base.action_space
    assert wrapped.initial_pos is None
    assert wrapped.ignore_actions == set()

    init_act = base.action_space.sample()
    action_scaler = base.generate_chunk / base.sample_rate
    wrapped = ActionByAcceleration(base, action_scaler, initial_pos=init_act, ignore_actions=[ASN.TONGUE_INDEX])
    assert wrapped.action_scaler == action_scaler
    assert wrapped.initial_pos is init_act
    assert wrapped.ignore_actions == set([ASN.TONGUE_INDEX])


def test_convert_space_to_acceleration():
    cls = ActionByAcceleration

    b = spaces.Box(-1, 1)
    f = cls.convert_space_to_acceleration
    assert_space(f(b), b)
    assert_space(f(spaces.Box(0, 1)), spaces.Box(-0.5, 0.5))
    assert_space(f(spaces.Box(-10, 2, dtype=int)), spaces.Box(-6, 6, dtype=int))
    assert_space(f(spaces.Box(0.0, 5.0, shape=(10, 2))), spaces.Box(-2.5, 2.5, shape=(10, 2)))
    assert_space(f(spaces.Box(0, math.inf)), spaces.Box(-math.inf, math.inf))
    assert_space(f(spaces.Box(-math.inf, 0)), spaces.Box(-math.inf, math.inf))
    assert_space(f(spaces.Box(-math.inf, math.inf)), spaces.Box(-math.inf, math.inf))
    assert_space(f(spaces.Box(1, 1)), spaces.Box(0, 0))

    # Gym API Error
    # assert_space(f(spaces.Box(math.inf, math.inf)), spaces.Box(0.0, 0.0))
    # Box(math.inf, math.inf) -> Box(-inf, inf, (1,), float32) !!!!!!


def test_define_action_space():
    base = PynkTrombone(target_sound_files)

    action_scaler = base.generate_chunk / base.sample_rate
    wrapped = ActionByAcceleration(base, action_scaler)
    base_acts = base.action_space
    wrapped_acts = wrapped.action_space
    for k in base_acts.keys():
        assert_space(wrapped_acts[k], wrapped.convert_space_to_acceleration(base_acts[k]))  # type: ignore

    ignore_actions = set([ASN.TONGUE_DIAMETER, ASN.TONGUE_INDEX])
    wrapped = ActionByAcceleration(base, action_scaler, ignore_actions=ignore_actions)
    wrapped_acts = wrapped.action_space
    for k in base_acts.keys():
        if k not in ignore_actions:
            assert_space(wrapped_acts[k], wrapped.convert_space_to_acceleration(base_acts[k]))  # type: ignore
        else:
            assert_space(wrapped_acts[k], base_acts[k])  # type: ignore


def initialize_state():
    base = PynkTrombone(target_sound_files)
    action_scaler = base.generate_chunk / base.sample_rate
    wrapped = ActionByAcceleration(base, action_scaler, copy.deepcopy(initial_pos))

    # called at __init__
    assert_dict_space_key(wrapped.velocities, ASN)
    assert_dict_space_key(wrapped.positions, ASN)

    assert initial_pos is wrapped.initial_pos
    assert initial_pos is not wrapped.positions

    velocities = wrapped.velocities
    for k in initial_pos.keys():
        assert np.all(velocities[k] == 0.0)

    # without initial pos
    wrapped = ActionByAcceleration(base, action_scaler)
    assert_dict_space_key(wrapped.positions, ASN)


def test_reset():
    base = PynkTrombone(target_sound_files)
    action_scaler = base.generate_chunk / base.sample_rate
    wrapped = ActionByAcceleration(base, action_scaler)

    pos0 = wrapped.positions
    vel0 = wrapped.velocities

    wrapped.reset()
    pos1 = wrapped.positions
    vel1 = wrapped.velocities

    assert pos0 is not pos1
    assert vel0 is not vel1


def test_action():
    base = PynkTrombone(target_sound_files)
    action_scaler = 1.0
    wrapped = ActionByAcceleration(base, action_scaler)

    initial_poses = copy.deepcopy(wrapped.positions)
    initial_vels = copy.deepcopy(wrapped.velocities)
    action = wrapped.action_space.sample()

    out_pos = wrapped.action(action)
    current_vels = wrapped.velocities
    current_poses = wrapped.positions

    assert out_pos is not action

    for k in action.keys():
        act = action[k] * action_scaler
        vel = initial_vels[k]
        pos = initial_poses[k]
        pos_space: spaces.Box = wrapped.position_space[k]  # type: ignore

        vel += act
        pos += vel
        is_limit = np.logical_or(pos < pos_space.low, pos_space.high < pos)
        vel[is_limit] = 0
        pos = np.clip(pos, pos_space.low, pos_space.high)

        assert np.all(pos == current_poses[k])
        assert np.all(vel == current_vels[k])
        cp = current_poses[k]
        assert np.all(np.logical_and(cp >= pos_space.low, cp <= pos_space.high))

    action_scaler = 0.5
    wrapped = ActionByAcceleration(base, action_scaler, copy.deepcopy(initial_pos))

    out_pos = wrapped.action(example_action)
    assert abs(out_pos[ASN.PITCH_SHIFT].item() - 0.05) < 1e-8
    assert abs(out_pos[ASN.TENSENESS].item() - 0.1) < 1e-8
    assert abs(out_pos[ASN.TRACHEA].item() - 0.45) < 1e-8
    assert abs(out_pos[ASN.EPIGLOTTIS].item() - 0.55) < 1e-8
    assert abs(out_pos[ASN.VELUM].item() - 0.0) < 1e-8
    assert abs(out_pos[ASN.TONGUE_INDEX].item() - 19.0) < 1e-8
    assert abs(out_pos[ASN.TONGUE_DIAMETER].item() - 1.5) < 1e-8
    assert abs(out_pos[ASN.LIPS].item() - 1.5) < 1e-8

    ignore_actions = set([ASN.EPIGLOTTIS, ASN.PITCH_SHIFT])
    wrapped = ActionByAcceleration(base, action_scaler, copy.deepcopy(initial_pos), ignore_actions)
    out_pos = wrapped.action(example_action)
    assert abs(out_pos[ASN.EPIGLOTTIS].item() + 1.1) < 1e-8
    assert abs(out_pos[ASN.PITCH_SHIFT].item() - 0.1) < 1e-8
