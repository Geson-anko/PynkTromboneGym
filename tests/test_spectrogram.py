from pynktrombonegym import spectrogram as spct


def test_calc_rfft_channel_num():
    f = spct.calc_rfft_channel_num
    assert f(1024) == 513
    assert f(256) == 129
    assert f(400) == 201
    assert f(501) == 251
