from pynktrombonegym import spaces as ptspaces


def test_BaseSpace():
    bs = ptspaces.BaseSpace()

    d = bs.to_dict()
    assert type(d) is dict
    assert d == dict()
    try:
        ptspaces.BaseSpace.from_dict({"a": 0})
        raise AssertionError
    except TypeError:
        pass
