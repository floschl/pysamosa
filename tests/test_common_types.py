from pysamosa.common_types import (
    FittingSettings,
    L1bSourceType,
    ModelSettings,
    SensorType,
    WaveformSettings,
)


def test_fitting_sets():
    fit_sets = FittingSettings()
    assert fit_sets.Fit_Var_2_Init_Hs == 2.0


def test_waveform_sets():
    wf_sets = WaveformSettings()
    assert wf_sets.zp_oversampling_factor == 1
    assert wf_sets.np == 128

    wf_sets = WaveformSettings(np=256, zp_oversampling_factor=2)
    assert wf_sets.zp_oversampling_factor == 2
    assert wf_sets.np == 512

    wf_sets = WaveformSettings(np=256, zp_oversampling_factor=2)
    assert wf_sets.zp_oversampling_factor == 2
    assert wf_sets.np == 512

    wf_sets = WaveformSettings(internal_oversampling_factor=4)
    assert wf_sets.np == 512

    wf_sets = WaveformSettings(internal_oversampling_factor=4, zp_oversampling_factor=2)
    assert wf_sets.np == 1024

    # defaults
    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.GPOD)
    assert wf_sets.np == 512
    assert wf_sets.zp_oversampling_factor == 2

    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3)
    assert wf_sets.np == 128
    assert wf_sets.zp_oversampling_factor == 1

    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S6_F04)
    assert wf_sets.np == 512
    assert wf_sets.zp_oversampling_factor == 2

    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S6_F06)
    assert wf_sets.np == 512
    assert wf_sets.zp_oversampling_factor == 2

    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S6_F04_FFSAR)
    assert wf_sets.np == 512
    assert wf_sets.zp_oversampling_factor == 2

    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S6_F06_FFSAR)
    assert wf_sets.np == 512
    assert wf_sets.zp_oversampling_factor == 2

    wf_sets = WaveformSettings.get_default_src_type(
        L1bSourceType.EUM_S6_F04, internal_oversampling_factor=2
    )
    assert wf_sets.np == 1024
    assert wf_sets.zp_oversampling_factor == 2

    wf_sets = WaveformSettings.get_default_src_type(
        L1bSourceType.EUM_S3, zp_oversampling_factor=2
    )
    assert wf_sets.zp_oversampling_factor == 2
    assert wf_sets.np == 256


def test_model_sets():
    ms = ModelSettings.get_default_sets(st=SensorType.S3)
    assert ms.alpha_p_mean == 0.5

    ms = ModelSettings.get_default_sets(st=SensorType.S6_F04)
    assert ms.alpha_p_mean == 0.55
