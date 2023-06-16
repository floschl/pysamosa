import matplotlib.pyplot as plt
import pytest

from pysamosa.common_types import (
    L1bSourceType,
    ModelSettings,
    SensorSettings,
    SensorType,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.l1b_simulator import L1bSimulator


@pytest.mark.parametrize(
    "swh",
    # np.arange(1, 10, 2),
    [2.0],
)
def test_gen_l1b_data_single(swh):
    n_realisations = 3
    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3)
    sensor_sets = SensorSettings.get_default_sets(st=SensorType.S3)

    model_sets = ModelSettings.get_default_sets(
        st=sensor_sets.sensor_type,
        wf_sets=wf_sets,
    )

    l1b_sim = L1bSimulator(
        model_sets=model_sets,
        swh=swh,
        Pu=1.0,
        sensor_sets=sensor_sets,
        wf_sets=wf_sets,
        settings_preset=SettingsPreset.NONE,
    )
    l1b_sim_it = iter(l1b_sim)

    for i in range(n_realisations):
        l1b_data_single = next(l1b_sim_it)
        plt.plot(l1b_data_single["wf"], linewidth=0.7)
        assert l1b_data_single["wf"] is not None

    plt.title(f"SWH={swh}m")
    plt.grid()
    plt.show()


@pytest.mark.parametrize(
    "swh",
    # np.arange(1, 10, 2),
    [2.0],
)
def test_gen_l1b_data_add_interference(swh):
    n_realisations = 3
    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3)
    sensor_sets = SensorSettings.get_default_sets(st=SensorType.S3)

    model_sets = ModelSettings.get_default_sets(
        st=sensor_sets.sensor_type,
        wf_sets=wf_sets,
    )

    l1b_sim = L1bSimulator(
        model_sets=model_sets,
        swh=swh,
        Pu=1.0,
        sensor_sets=SensorSettings(),
        add_interference=True,
        wf_sets=WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3),
        settings_preset=SettingsPreset.NONE,
    )
    l1b_sim_it = iter(l1b_sim)

    for i in range(n_realisations):
        l1b_data_single = next(l1b_sim_it)
        plt.plot(l1b_data_single["wf"], linewidth=0.7)
        assert l1b_data_single["wf"] is not None

    plt.title(f"SWH={swh}m")
    plt.grid()
    plt.show()
