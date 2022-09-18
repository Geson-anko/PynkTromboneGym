import math

from gym import spaces

from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.wrappers.action_by_acceleration import ActionByAcceleration

from ..test_env import assert_space, target_sound_files


def test__init__():
    base = PynkTrombone(target_sound_files)

    action_scaler = 1.0
    wrapped = ActionByAcceleration(base, action_scaler)

    assert wrapped.action_scaler == action_scaler
    assert wrapped.position_space is base.action_space
    assert wrapped.initial_pos is None

    init_act = base.action_space.sample()
    action_scaler = base.generate_chunk / base.sample_rate
    wrapped = ActionByAcceleration(base, action_scaler, initial_pos=init_act)
    assert wrapped.action_scaler == action_scaler
    assert wrapped.initial_pos is init_act


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
