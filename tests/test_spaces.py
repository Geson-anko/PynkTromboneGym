from pynktrombonegym import spaces as ptspaces


def assert_space_names(name_cls: object):
    for k in name_cls.__dict__.keys():
        if not k.startswith("__"):
            assert k.isupper(), "Key name must be upper case because it is constant value."
            assert k == getattr(name_cls, k).upper()


def test_ObservationSpaceNames():
    name_cls = ptspaces.ObservationSpaceNames
    assert_space_names(name_cls)

    assert name_cls.TARGET_SOUND_WAVE == "target_sound_wave"
    assert name_cls.GENERATED_SOUND_WAVE == "generated_sound_wave"
    assert name_cls.TARGET_SOUND_SPECTROGRAM == "target_sound_spectrogram"
    assert name_cls.GENERATED_SOUND_SPECTROGRAM == "generated_sound_spectrogram"
    assert name_cls.FREQUENCY == "frequency"
    assert name_cls.PITCH_SHIFT == "pitch_shift"
    assert name_cls.TENSENESS == "tenseness"
    assert name_cls.CURRENT_TRACT_DIAMETERS == "current_tract_diameters"
    assert name_cls.NOSE_DIAMETERS == "nose_diameters"


def test_ActionSpaceNames():
    name_cls = ptspaces.ActionSpaceNames
    assert_space_names(name_cls)

    assert name_cls.PITCH_SHIFT == "pitch_shift"
    assert name_cls.TENSENESS == "tenseness"
    assert name_cls.TRACHEA == "trachea"
    assert name_cls.EPIGLOTTIS == "epiglottis"
    assert name_cls.VELUM == "velum"
    assert name_cls.TONGUE_INDEX == "tongue_index"
    assert name_cls.TONGUE_DIAMETER == "tongue_diameter"
    assert name_cls.LIPS == "lips"
