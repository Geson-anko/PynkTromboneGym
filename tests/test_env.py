import glob
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
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
    assert default.generate_chunk == 1024
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

    assert "target_sound_wave" in obsk
    assert "generated_sound_wave" in obsk
    assert "target_sound_spectrogram" in obsk
    assert "generated_sound_spectrogram" in obsk
    assert "frequency" in obsk
    assert "pitch_shift" in obsk
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


def test_property_target_sound_wave():
    default = env.PynkTrombone(target_sound_files)
    tgt_full = default.target_sound_wave_full
    tgt0 = tgt_full[: default.generate_chunk]
    assert np.all(default.target_sound_wave == tgt0)

    default.current_step = 1
    tgt1 = tgt_full[default.generate_chunk : 2 * default.generate_chunk]
    assert np.all(default.target_sound_wave == tgt1)


def test_property_generated_sound_wave():
    default = env.PynkTrombone(target_sound_files)
    assert default.current_step == 0  # Assumption
    assert len(default.generated_sound_wave) == default.generate_chunk


def test_set_target_sound_files():
    default = env.PynkTrombone(target_sound_files)

    file_paths = ["aaa", "bbb", "ccc"]
    assert default.target_sound_files == target_sound_files
    default.set_target_sound_files(file_paths)
    assert default.target_sound_files == file_paths


def test_initialize_state():
    default = env.PynkTrombone(target_sound_files)
    default.initialize_state()

    assert default.current_step == 0
    assert type(default.target_sound_wave_full) is np.ndarray
    assert np.all(default._generated_sound_wave_2chunks == 0.0)
    assert default._stored_state_figures == []

    voc0 = default.voc
    default.current_step += 10  # Assumption
    default.initialize_state()
    voc1 = default.voc
    assert voc0 is not voc1
    assert default.current_step == 0


def test_define_action_space():
    default = env.PynkTrombone(target_sound_files)
    acts = default.define_action_space()

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
    obs = default.define_observation_space()

    spct_shape = (
        spct.calc_rfft_channel_num(default.stft_window_size),
        spct.calc_target_sound_spectrogram_length(
            default.generate_chunk, default.stft_window_size, default.stft_hop_length
        ),
    )
    chunk = default.generate_chunk
    assert_space(obs["target_sound_wave"], spaces.Box(-1.0, 1.0, (chunk,)))
    assert_space(obs["generated_sound_wave"], spaces.Box(-1.0, 1.0, (chunk,)))
    assert_space(obs["target_sound_spectrogram"], spaces.Box(0, float("inf"), spct_shape))
    assert_space(obs["generated_sound_spectrogram"], spaces.Box(0, float("inf"), spct_shape))
    assert_space(obs["frequency"], spaces.Box(0, default.sample_rate // 2))
    assert_space(obs["pitch_shift"], spaces.Box(-1.0, 1.0))
    assert_space(obs["tenseness"], spaces.Box(0.0, 1.0))
    assert_space(obs["current_tract_diameters"], spaces.Box(0.0, 5.0, (default.voc.tract_size,)))
    assert_space(obs["nose_diameters"], spaces.Box(0.0, 5.0, (default.voc.nose_size,)))


def test_define_reward_range():
    default = env.PynkTrombone(target_sound_files)
    reward_range = default.define_reward_range()

    assert reward_range == (-float("inf"), 0.0)


def test_load_sound_wave_randomly():
    default = env.PynkTrombone(target_sound_files)
    wave = default.load_sound_wave_randomly()

    assert type(wave) is np.ndarray
    assert len(wave.shape) == 1


def test_get_target_sound_spectrogram():
    default = env.PynkTrombone(target_sound_files)
    assert default.current_step == 0
    assert type(default.target_sound_wave_full) is np.ndarray
    assert len(default.target_sound_wave_full.shape) == 1

    spect = default.get_target_sound_spectrogram()

    length = spct.calc_target_sound_spectrogram_length(
        default.generate_chunk, default.stft_window_size, default.stft_hop_length
    )
    channel = spct.calc_rfft_channel_num(default.stft_window_size)

    assert type(spect) is np.ndarray
    assert spect.dtype == np.float32
    assert spect.shape == (channel, length)
    assert np.min(spect) >= 0.0


def test_get_generated_sound_spectrogram():
    default = env.PynkTrombone(target_sound_files)
    assert type(default._generated_sound_wave_2chunks) is np.ndarray
    w2c = default._generated_sound_wave_2chunks
    assert len(w2c) == 2 * default.generate_chunk

    spect = default.get_generated_sound_spectrogram()

    length = spct.calc_target_sound_spectrogram_length(
        default.generate_chunk, default.stft_window_size, default.stft_hop_length
    )
    channel = spct.calc_rfft_channel_num(default.stft_window_size)

    assert type(spect) is np.ndarray
    assert spect.dtype == np.float32
    assert spect.shape == (channel, length)
    assert np.min(spect) >= 0.0


def test_get_current_observation():
    dflt = env.PynkTrombone(target_sound_files)
    dflt.initialize_state()

    # Type checking
    obs_dict = dflt.get_current_observation()
    obs = env.ObservationSpace.from_dict(obs_dict)
    sample_obs = dflt.observation_space.sample()
    for sk in sample_obs.keys():
        ov, sv = obs_dict[sk], sample_obs[sk]
        assert type(ov) == type(sv), sk
        assert ov.shape == sv.shape, sk
        assert ov.dtype == sv.dtype, sk

    assert isinstance(obs.target_sound_wave, np.ndarray)
    assert isinstance(obs.generated_sound_wave, np.ndarray)
    assert isinstance(obs.target_sound_spectrogram, np.ndarray)
    assert isinstance(obs.generated_sound_spectrogram, np.ndarray)
    assert isinstance(obs.frequency, np.ndarray)
    assert isinstance(obs.pitch_shift, np.ndarray)
    assert isinstance(obs.tenseness, np.ndarray)
    assert isinstance(obs.current_tract_diameters, np.ndarray)
    assert isinstance(obs.nose_diameters, np.ndarray)

    # Value checking
    assert np.all(obs.target_sound_wave == dflt.target_sound_wave)
    assert np.all(obs.generated_sound_wave == dflt.generated_sound_wave)
    assert np.all(obs.target_sound_spectrogram == dflt.get_target_sound_spectrogram())
    assert np.all(obs.generated_sound_spectrogram == dflt.get_generated_sound_spectrogram())
    assert obs.frequency.item() == dflt.voc.frequency

    pitch_shift = np.log2(dflt.voc.frequency / dflt.default_frequency)
    assert abs(obs.pitch_shift.item() - pitch_shift) < 1e-10
    assert abs(obs.tenseness.item() - dflt.voc.tenseness) < 1e-6
    assert np.all(obs.current_tract_diameters == dflt.voc.current_tract_diameters.astype(np.float32))
    assert np.all(obs.nose_diameters == dflt.voc.nose_diameters.astype(np.float32))


def test_reset():
    dflt = env.PynkTrombone(target_sound_files)

    obs = dflt.reset()
    assert isinstance(obs, OrderedDict)
    env.ObservationSpace.from_dict(obs)


def test_compute_reward():
    dflt = env.PynkTrombone(target_sound_files)

    dflt.reset()
    r = dflt.compute_reward()
    assert r <= 0
    dflt.reset()
    r = dflt.compute_reward()
    assert r <= 0
    dflt.reset()
    r = dflt.compute_reward()
    assert r <= 0


def test_property_done():
    dflt = env.PynkTrombone(target_sound_files)

    dflt.reset()
    assert not dflt.done
    dflt.reset()
    assert not dflt.done
    dflt.reset()
    assert not dflt.done

    c = dflt.generate_chunk
    le = len(dflt.target_sound_wave_full)
    max_step = math.ceil(le / c)
    dflt.current_step = max_step
    assert dflt.done
    dflt.current_step = max_step - 1
    assert not dflt.done
    dflt.current_step = max_step + 1
    assert dflt.done


def test_property_max_steps():
    dflt = env.PynkTrombone(target_sound_files)

    for _ in range(10):
        dflt.reset()
        m = math.ceil(len(dflt.target_sound_wave_full) / dflt.generate_chunk)
        assert dflt.max_steps == m


def test_step():
    dflt = env.PynkTrombone(target_sound_files)

    dflt.reset()
    act = dflt.action_space.sample()
    obs, reward, done, info = dflt.step(act)

    acts = env.ActionSpace.from_dict(act)
    # obss = env.ObservationSpace.from_dict(obs)

    voc = dflt.voc
    assert voc.frequency == dflt.default_frequency * (2**acts.pitch_shift)
    assert voc.tenseness == acts.tenseness
    assert abs(voc.tract.trachea - acts.trachea) < 1e-10
    assert abs(voc.tract.epiglottis - acts.epiglottis) < 1e-10
    assert abs(voc.velum - acts.velum) < 1e-10
    # Missing assertion of tongue_index
    # Missing assertion of tongue_diameter
    assert abs(voc.tract.lips - acts.lips) < 1e-10
    assert dflt.current_step == 1

    ### Step until done.
    dflt.reset()
    max_steps = dflt.max_steps
    for i in range(max_steps):
        act = dflt.action_space.sample()
        obs, reward, done, info = dflt.step(act)
        env.ObservationSpace.from_dict(obs)
        assert isinstance(info, dict)
        assert isinstance(reward, float)
        if (i + 1) == max_steps:
            assert done
        else:
            assert not done

    try:
        dflt.step(act)
        raise AssertionError
    except RuntimeError:
        pass


def test_create_state_figure():
    dflt = env.PynkTrombone(target_sound_files)
    dflt.reset()
    figure = dflt.create_state_figure()
    figure.savefig(f"data/test_results/{__name__}.test_create_state_figure.png")


def test_fig2argb_array():
    dflt = env.PynkTrombone(target_sound_files)
    dflt.reset()
    figure = dflt.create_state_figure()
    array = env.fig2rgba_array(figure)

    assert isinstance(array, np.ndarray)
    assert array.shape[-1] == 4
    assert array.dtype == np.uint8
    assert array.ndim == 3

    plt.imsave(f"data/test_results/{__name__}.test_fig2rgba_array.png", array)


def test_render():
    dflt = env.PynkTrombone(target_sound_files)
    dflt.reset()
    assert dflt.render(None) is None
    figure = dflt.render("single_figure")
    assert isinstance(figure, plt.Figure)
    figure.savefig(f"data/test_results/{__name__}.test_render.figure.png")

    array = dflt.render("single_rgb_array")
    assert isinstance(array, np.ndarray)
    assert array.shape[-1] == 3
    plt.imsave(f"data/test_results/{__name__}.test_render.single_rgb_array.png", array)
    plt.close()

    dflt.reset()
    render_times = 5
    for _ in range(render_times):
        dflt.render()
    figures = dflt.render("figures")
    assert len(figures) == render_times + 1
    assert dflt._stored_state_figures == []
    for f in figures:
        assert isinstance(f, plt.Figure)
    plt.close()

    for _ in range(render_times):
        dflt.render()
    fig_arrays = dflt.render("rgb_arrays")
    assert len(fig_arrays) == render_times + 1
    assert dflt._stored_state_figures == []
    for fa in fig_arrays:
        assert isinstance(fa, np.ndarray)
        assert fa.shape[-1] == 3
        assert fa.ndim == 3
    plt.close()

    try:
        dflt.render("1234567890")  # type: ignore
        raise AssertionError
    except NotImplementedError:
        pass


def test_mean_squared_error():
    f = env.mean_squared_error

    o1 = np.zeros(10)
    t1 = np.ones(10)
    assert isinstance(f(o1, t1), float)
    assert abs(f(o1, t1) - 1.0) < 1e-10

    o2 = np.arange(4, dtype=float)
    t2 = np.arange(-4, 0, dtype=float)
    assert abs(f(o2, t2) - 16.0) < 1e-10
