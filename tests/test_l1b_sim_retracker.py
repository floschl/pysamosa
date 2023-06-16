import matplotlib.pyplot as plt

from pysamosa import retracker, simple_logger
from pysamosa.common_types import (
    SENSOR_SETS_DEFAULT_S3,
    L1bSourceType,
    ModelSettings,
    SensorSettings,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.data_access import get_model_param_obj_from_l1b_data
from pysamosa.l1b_simulator import L1bSimulator
from pysamosa.settings_manager import get_default_base_settings
from tests.helpers import plot_retrack_result

l1b_src_type, settings_preset = (
    L1bSourceType.EUM_S3,
    SettingsPreset.CORALv1,
)
rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(
    settings_preset=settings_preset,
    l1b_src_type=l1b_src_type,
)

# retrack_sets.leading_edge_weight_factor = 1.0


def test_retrack_l1bsim_w_interference():
    n_realisations = 5
    swh = 2.5

    model_sets = ModelSettings.get_default_sets(
        st=sensor_sets.sensor_type,
        wf_sets=wf_sets,
    )

    l1b_sim = L1bSimulator(
        model_sets=model_sets,
        swh=swh,
        Pu=1.0,
        sensor_sets=SENSOR_SETS_DEFAULT_S3,
        add_thermal_speckle_noise=True,
        add_interference=True,
        wf_sets=WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3),
        settings_preset=SettingsPreset.NONE,
    )
    l1b_sim_it = iter(l1b_sim)

    # run L1B simulations and retracking
    for i in range(n_realisations):
        fig, ax = plt.subplots()
        l1b_data_single = next(l1b_sim_it)
        model_params = get_model_param_obj_from_l1b_data(l1b_data_single, ind=0)

        simple_logger.set_root_logger()

        sr = retracker.SamosaRetracker(
            retrack_sets=retrack_sets,
            fitting_sets=fitting_sets,
            wf_sets=wf_sets,
            sensor_sets=sensor_sets,
        )

        # start fitting
        res_fit = sr.fit_wf(l1b_data_single=l1b_data_single, model_params=model_params)

        # plot fitted curve
        l2_data_single_sim = l1b_sim.get_l2_data_single()
        plot_retrack_result(
            l1b_data_single,
            l2_data_single_sim,
            res_fit,
            model_params=model_params,
            model_sets=sr.model_sets,
            retrack_sets=retrack_sets,
            wf_sets=wf_sets,
            sensor_sets=sensor_sets,
            ax=ax,
            show_l2_model=False,
        )

        fig.suptitle(
            f'Simulated L1B-waveform, SWH={l2_data_single_sim["swh"]}m', fontsize=8
        )
        fig.show()


def test_retrack_l1bsim_wo_interference():
    n_realisations = 1
    swh = 2.5
    # swh = 17

    model_sets = ModelSettings.get_default_sets(
        st=sensor_sets.sensor_type,
        wf_sets=wf_sets,
    )

    l1b_sim = L1bSimulator(
        model_sets=model_sets,
        swh=swh,
        Pu=1.0,
        sensor_sets=SensorSettings(),
        add_thermal_speckle_noise=True,
        # add_thermal_speckle_noise=False,
        add_interference=False,
        wf_sets=WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3),
        settings_preset=SettingsPreset.NONE,
    )
    l1b_sim_it = iter(l1b_sim)

    # run L1B simulations and retracking
    for i in range(n_realisations):
        fig, ax = plt.subplots()
        l1b_data_single = next(l1b_sim_it)
        model_params = get_model_param_obj_from_l1b_data(l1b_data_single, ind=0)

        simple_logger.set_root_logger()

        sr = retracker.SamosaRetracker(
            retrack_sets=retrack_sets,
            fitting_sets=fitting_sets,
            wf_sets=wf_sets,
            sensor_sets=SENSOR_SETS_DEFAULT_S3,
        )

        # start fitting
        res_fit = sr.fit_wf(l1b_data_single=l1b_data_single, model_params=model_params)

        # plot fitted curve
        l2_data_single_sim = l1b_sim.get_l2_data_single()
        plot_retrack_result(
            l1b_data_single,
            l2_data_single_sim,
            res_fit,
            model_params=model_params,
            model_sets=sr.model_sets,
            retrack_sets=retrack_sets,
            wf_sets=wf_sets,
            sensor_sets=SensorSettings(),
            ax=ax,
            show_l2_model=False,
        )

        fig.subplots_adjust(top=0.7)
        fig.suptitle(
            f'Simulated L1B-waveform, SWH={l2_data_single_sim["swh"]}m', fontsize=8
        )
        fig.show()
