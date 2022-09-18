from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.wrappers.action_by_acceleration import ActionByAcceleration

from ..test_env import target_sound_files


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
