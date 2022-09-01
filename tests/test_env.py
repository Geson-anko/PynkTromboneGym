import glob

from gym import spaces

from pynktrombonegym import env
from pynktrombonegym import spectrogram as spct

target_sound_files = glob.glob("data/sample_target_sounds/*.wav")


def assert_space(space: spaces.Space, currect_space: spaces.Space):
    assert space == currect_space
    assert space.dtype == currect_space.dtype


def test__init__():

    ## test attributes.
    default = env.PynkTrombone(target_sound_files)
    assert default.target_sound_files == target_sound_files
    assert default.sample_rate == 44100
    assert default.default_frequency == 400
    assert default.generate_chunk == 512
    assert default.stft_window_size == 1024
    assert default.stft_hop_length == 256

    sample_rate = 44100
    default_frequency = 440
    generate_chunk = 512
    stft_window_size = 512
    stft_hop_length = 256
    mod = env.PynkTrombone(
        target_sound_files, sample_rate, default_frequency, generate_chunk, stft_window_size, stft_hop_length
    )
    assert mod.target_sound_files == target_sound_files
    assert mod.sample_rate == sample_rate
    assert mod.default_frequency == default_frequency
    assert mod.generate_chunk == generate_chunk
    assert mod.stft_window_size == stft_window_size
    assert mod.stft_hop_length == stft_hop_length

    # test define spaces
    rr = default.reward_range
    obss = default.observation_space
    acts = default.action_space

    assert type(rr) is tuple
    assert type(obss) is spaces.Dict
    assert type(acts) is spaces.Dict

    obsk = list(obss.keys())
    actk = list(acts.keys())

    assert rr == (-float("inf"), 0.0)

    assert "target_sound" in obsk
    assert "previous_generated_sound" in obsk
    assert "current_frequency" in obsk
    assert "current_pitch_shift" in obsk
    assert "tenseness" in obsk
    assert "current_tract_diameters" in obsk
    assert "nose_diameters" in obsk

    assert "pitch_shift" in actk
    assert "tenseness" in actk
    assert "trachea" in actk
    assert "epiglottis" in actk
    assert "velum" in actk
    assert "tongue_index" in actk
    assert "tongue_diameter" in actk
    assert "lips" in actk


def test_set_target_sound_files():
    default = env.PynkTrombone(target_sound_files)

    file_paths = ["aaa", "bbb", "ccc"]
    assert default.target_sound_files == target_sound_files
    default.set_target_sound_files(file_paths)
    assert default.target_sound_files == file_paths


def test_define_action_space():
    default = env.PynkTrombone(target_sound_files)
    default.define_action_space()

    acts = default.action_space
    assert_space(acts["pitch_shift"], spaces.Box(-1.0, 1.0))
    assert_space(acts["tenseness"], spaces.Box(0.0, 1.0))
    assert_space(acts["trachea"], spaces.Box(0, 3.5))
    assert_space(acts["epiglottis"], spaces.Box(0, 3.5))
    assert_space(acts["velum"], spaces.Box(0, 3.5))
    assert_space(acts["tongue_index"], spaces.Box(12, 40, dtype=int))
    assert_space(acts["tongue_diameter"], spaces.Box(0, 3.5))
    assert_space(acts["lips"], spaces.Box(0, 1.5))


def test_define_observation_space():
    default = env.PynkTrombone(target_sound_files)
    default.define_observation_space()
    obs = default.observation_space

    spct_shape = (
        spct.calc_rfft_channel_num(default.stft_window_size),
        spct.calc_target_sound_spectrogram_length(
            default.generate_chunk, default.stft_window_size, default.stft_hop_length
        ),
    )

    assert_space(obs["target_sound"], spaces.Box(0, float("inf"), spct_shape))
    assert_space(obs["previous_generated_sound"], spaces.Box(0, float("inf"), spct_shape))
    assert_space(obs["current_frequency"], spaces.Box(0, default.sample_rate // 2))
    assert_space(obs["current_pitch_shift"], spaces.Box(-1.0, 1.0))
    assert_space(obs["tenseness"], spaces.Box(0.0, 1.0))
    assert_space(obs["current_tract_diameters"], spaces.Box(0.0, 5.0, (default.voc.tract_size,)))
    assert_space(obs["nose_diameters"], spaces.Box(0.0, 5.0, (default.voc.nose_size,)))


def test_define_reward_range():
    default = env.PynkTrombone(target_sound_files)
    default.define_reward_range()

    assert default.reward_range == (-float("inf"), 0.0)
