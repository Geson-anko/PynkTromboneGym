"""Tools for spectrogram operatings."""


def calc_rfft_channel_num(n_fft: int) -> int:
    """calculate output channel number of rfft operation at `n_fft`.

    Args:
        n_fft (int): wave length of fft.

    Returns:
        rfft_channel_num (int): Channel length of rfft output.
    """
    return int(n_fft / 2) + 1
