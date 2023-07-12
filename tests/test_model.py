import matplotlib.pyplot as plt
import numpy as np

from pysamosa.common_types import (
    L1bSourceType,
    SensorSettings,
    SensorType,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.model import ModelParameter, ModelSettings, SamosaModel
from pysamosa.retracker import oversample_wf


def test_setting_model_parameter_obj():
    mp = ModelParameter(ascending=True)
    print(mp)
    assert mp
    assert mp.alt_m == 815000


def test_waveform_single_look():
    models_sets = ModelSettings.get_default_sets(st=SensorType.S3)
    sm = SamosaModel(
        sensor_sets=SensorSettings(),
        wf_sets=WaveformSettings(),
        model_sets=models_sets,
        settings_preset=SettingsPreset.NONE,
    )

    # swh = np.arange(11) + 0.5
    swh = [0.25, 0.75, 1.0, 2.0, 5, 7, 9, 11]
    P_u = 1
    t_0 = -16

    for s in swh:
        wf = sm.get_waveform_singlelook(
            Pu=P_u, Hs=s, t0_ns=t_0, fa_Hz=0.0, model_params=ModelParameter()
        )
        # wf = wf + np.random.randn(len(wf)) * 0.01 * np.nanmax(wf)  # NOISE ADDITION (not in the SAMOSA documentation )
        # plt.plot(wf / np.nanmax(wf), label=f'swh={s}', linewidth=0.7)
        plt.plot(wf, label=f"swh={s}", linewidth=0.7)

    plt.legend()
    plt.grid()
    plt.title("SAMOSA single-look model")
    plt.ylabel("normalised power")
    plt.xlabel("# range bin")
    plt.show()


def test_multilooked_oversample():
    plt.figure(dpi=600)

    for oversampling_factor, m in zip([1, 4], ["x", None]):
        wf_sets = WaveformSettings.get_default_src_type(
            L1bSourceType.EUM_S3, internal_oversampling_factor=oversampling_factor
        )
        model_sets = ModelSettings()
        sm = SamosaModel(
            model_sets=model_sets,
            wf_sets=wf_sets,
            sensor_sets=SensorSettings(),
            settings_preset=SettingsPreset.NONE,
        )

        swh = 3.0
        Pu = 1.0
        t_0_ns = 20
        nu = 0

        epoch_ref_gate = 64
        model_params = ModelParameter(epoch_ref_gate=epoch_ref_gate)
        wf = sm.get_waveform_multilook(
            Pu=Pu, Hs=swh, t0_ns=t_0_ns, nu=nu, model_params=model_params
        )

        # calculate interpolated x_vector to map all models on the same x-axis
        # scale
        x_interp = np.append(
            np.arange(0, 127 + 1 / oversampling_factor, 1 / oversampling_factor),
            np.arange(127 + 1 / oversampling_factor, 128, 1 / oversampling_factor),
        )

        plt.plot(
            x_interp,
            wf,
            linewidth=0.5,
            marker=m,
            markersize=0.8,
            label=f"wf_model Np={128*oversampling_factor}",
        )
        # plt.axvline(epoch_ref_gate, label=f'epoch_ref_gate (of={oversampling_factor})')

        assert not any(np.isnan(wf))

    plt.legend(fontsize=8)
    plt.xlabel("# range bin")
    plt.ylabel("normalised power")
    plt.grid()
    plt.show()


def test_different_model_len():
    plt.figure(dpi=600)

    oversampling_factor = 4

    for model_len, m in zip([509, 512, 500], ["x", None, "o"]):
        # for model_len, m in zip([512, 509],['x', None]):
        wf_sets = WaveformSettings.get_default_src_type(
            L1bSourceType.EUM_S3, zp_oversampling_factor=oversampling_factor
        )
        wf_sets.np = model_len
        model_sets = ModelSettings()
        sm = SamosaModel(
            model_sets=model_sets,
            wf_sets=wf_sets,
            sensor_sets=SensorSettings(),
            settings_preset=SettingsPreset.NONE,
        )

        swh = 3.0
        Pu = 1.0
        t_0_ns = 20
        nu = 0

        epoch_ref_gate = 64
        model_params = ModelParameter(
            epoch_ref_gate=epoch_ref_gate * oversampling_factor
        )
        wf = sm.get_waveform_multilook(
            Pu=Pu, Hs=swh, t0_ns=t_0_ns, nu=nu, model_params=model_params
        )
        # wf = sm.get_waveform_singlelook(Pu=Pu, Hs=swh, t0_ns=t_0_ns, nu=nu, model_params=model_params, fa_Hz=0)

        x_gates = np.arange(0, model_len)

        plt.plot(
            x_gates,
            wf,
            linewidth=0.5,
            marker=m,
            markersize=1,
            label=f"wf_model_len={model_len}",
        )
        # plt.axvline(epoch_ref_gate, label=f'epoch_ref_gate (of={oversampling_factor})')

    plt.legend(fontsize=8)
    plt.xlabel("# range bin")
    plt.ylabel("normalised power")
    plt.grid()
    plt.show()


def test_oversampled_model():
    plt.figure(dpi=600)

    swh = 3.0
    Pu = 1.0
    t_0_ns = 20
    nu = 0
    epoch_ref_gate = 64

    internal_oversampling_factor = 4

    # oversampling=1 and oversample by internal_oversampling_Factor
    sm_of1 = SamosaModel(
        model_sets=ModelSettings(),
        wf_sets=WaveformSettings.get_default_src_type(
            L1bSourceType.EUM_S3, internal_oversampling_factor=1
        ),
        sensor_sets=SensorSettings(),
        settings_preset=SettingsPreset.NONE,
    )
    model_params = ModelParameter(epoch_ref_gate=epoch_ref_gate)
    wf_of1 = sm_of1.get_waveform_multilook(
        Pu=Pu, Hs=swh, t0_ns=t_0_ns, nu=nu, model_params=model_params
    )

    wf_of1 = oversample_wf(wf_of1, internal_oversampling_factor)

    # oversampling=1 and oversample by internal_oversampling_Factor
    sm_of4 = SamosaModel(
        model_sets=ModelSettings(),
        wf_sets=WaveformSettings.get_default_src_type(
            L1bSourceType.EUM_S3,
            internal_oversampling_factor=internal_oversampling_factor,
            # np=509,
        ),
        sensor_sets=SensorSettings(),
        settings_preset=SettingsPreset.NONE,
    )
    # model_params = ModelParameter(epoch_ref_gate=epoch_ref_gate * internal_oversampling_factor)
    model_params = ModelParameter(epoch_ref_gate=epoch_ref_gate)
    wf_of4 = sm_of4.get_waveform_multilook(
        Pu=Pu, Hs=swh, t0_ns=t_0_ns, nu=nu, model_params=model_params
    )

    # wf_len = 128 * internal_oversampling_factor
    # int_oversampling_fac = sm_of4.wf_sets.internal_oversampling_factor
    # wf_len_interp = wf_len - (int_oversampling_fac-1)
    #
    # interpolator = Akima1DInterpolator(np.arange(wf_len), wf_of4)
    # # x_interp = np.arange(0, wf_len - 1 + 1/int_oversampling_fac, 1/int_oversampling_fac)
    # # x_interp = np.linspace(0, wf_len_interp, wf_len_interp + (int_oversampling_fac - 1))
    # # x_interp = np.arange(0, wf_len_interp)
    # x_interp = np.linspace(0, wf_len - 1, wf_len_interp)
    # # x_interp_app = np.append(x_interp, np.arange(127 + 1/int_oversampling_fac, 128, 1 / int_oversampling_fac))
    #
    # wf_oversampled = np.full(wf_len, np.nan)
    # wf_oversampled[:-(int_oversampling_fac-1)] = interpolator(x_interp)
    # wf_oversampled[-(int_oversampling_fac-1):] = wf_oversampled[-(int_oversampling_fac)]  #replace last (int_oversampling_factor - 1) values by the last valid ones
    # # wf_oversampled = interpolator(x_interp)
    # wf_of4 = wf_oversampled

    plt.plot(
        wf_of1,
        linewidth=0.3,
        marker="x",
        markersize=0.1,
        label=f"model.internal_of={1} + oversample",
    )
    plt.plot(
        wf_of4,
        linewidth=0.3,
        marker="o",
        markersize=0.3,
        label=f"model.internal_of={internal_oversampling_factor}",
    )
    # plt.axvline(epoch_ref_gate, label=f'epoch_ref_gate (of={oversampling_factor})')

    plt.legend(fontsize=8)
    plt.xlabel("# range bin")
    plt.ylabel("normalised power")
    plt.grid()
    plt.show()


def test_multilooked_vary_swh():
    oversampling_factor = 1
    wf_sets = WaveformSettings.get_default_src_type(
        L1bSourceType.EUM_S3, zp_oversampling_factor=oversampling_factor
    )
    model_sets = ModelSettings()
    sm = SamosaModel(
        model_sets=model_sets,
        wf_sets=wf_sets,
        sensor_sets=SensorSettings(),
        settings_preset=SettingsPreset.NONE,
    )

    swh = [0.25, 0.75, 1.0, 2.0, 5, 7, 9, 11]
    Pu = 1.0
    t_0 = 20
    nu = 0

    model_params = ModelParameter(epoch_ref_gate=64)
    for s in swh:
        wf = sm.get_waveform_multilook(
            Pu=Pu, Hs=s, t0_ns=t_0, nu=nu, model_params=model_params
        )
        # wf = wf + np.random.randn(len(wf)) * 0.01 * np.nanmax(wf)  # NOISE
        # ADDITION (not in the SAMOSA documentation )
        plt.plot(wf, label=f"swh={s}m", linewidth=1)

    plt.legend()
    plt.xlabel("# range bin")
    plt.ylabel("normalised power")
    plt.grid()
    plt.show()


def test_multilooked_vary_Pu():
    oversampling_factor = 1
    wf_sets = WaveformSettings.get_default_src_type(
        L1bSourceType.EUM_S3, zp_oversampling_factor=oversampling_factor
    )
    model_sets = ModelSettings()
    sm = SamosaModel(
        model_sets=model_sets,
        wf_sets=wf_sets,
        sensor_sets=SensorSettings(),
        settings_preset=SettingsPreset.NONE,
    )

    swh = 2.0
    Pu = np.arange(0, 5, 1.0)
    t_0 = 20
    nu = 0

    model_params = ModelParameter(epoch_ref_gate=64)
    for p in Pu:
        wf = sm.get_waveform_multilook(
            Pu=p, Hs=swh, t0_ns=t_0, nu=nu, model_params=model_params
        )
        # wf = wf + np.random.randn(len(wf)) * 0.01 * np.nanmax(wf)  # NOISE
        # ADDITION (not in the SAMOSA documentation )
        plt.plot(wf, label=f"Pu={p}m", linewidth=1)

    plt.legend()
    plt.xlabel("# range bin")
    plt.ylabel("normalised power")
    plt.grid()
    plt.show()
