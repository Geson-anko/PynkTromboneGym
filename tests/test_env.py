import glob

from pynktrombonegym import env

target_sound_files = glob.glob("data/sample_target_sounds/*.wav")


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


def test_set_target_sound_files():
    default = env.PynkTrombone(target_sound_files)

    file_paths = ["aaa", "bbb", "ccc"]
    assert default.target_sound_files == target_sound_files
    default.set_target_sound_files(file_paths)
    assert default.target_sound_files == file_paths
