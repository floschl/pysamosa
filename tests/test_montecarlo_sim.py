import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pysamosa.common_types import (
    L1bSourceType,
    ModelSettings,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.montecarlo_simulator import (
    CostFunctionType,
    MonteCarloSimulator,
    cost_functions,
    default_cost_functions,
    plot_cost_func_vs_swh,
    plot_swh_epoch_scatter,
)
from pysamosa.settings_manager import get_default_base_settings

settings_preset = (
    # SettingsPreset.NONE,
    SettingsPreset.CORALv1,
    # SettingsPreset.NONE,
    # SettingsPreset.NONE,
)
rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(
    settings_preset=settings_preset,
    l1b_src_type=L1bSourceType.EUM_S3,
)

model_sets = ModelSettings.get_default_sets(
    st=sensor_sets.sensor_type,
    wf_sets=wf_sets,
)

wf_sets_model = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3)


@pytest.fixture
def monte_sim():
    return MonteCarloSimulator(
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        model_sets=model_sets,
        wf_sets_retracker=wf_sets,
        wf_sets_model=wf_sets_model,
    )


def test_monte_carlo_sim_single_core(monte_sim):
    fit_res_list = []
    n_realisations = 2
    swhs = np.arange(1.0, 8.0, 1)
    for swh in swhs:
        fit_res_list.append(monte_sim(swh=swh, n_realisations=n_realisations))

    df = pd.concat(fit_res_list, ignore_index=True)

    assert df.shape[0] == (swhs.size * n_realisations)


def test_monte_carlo_sim_w_set_param(monte_sim):
    fit_res_list = []
    n_realisations = 2
    swhs = np.arange(1.0, 15.0, 1)
    for swh in swhs:
        fit_res_list.append(
            monte_sim(
                swh=swh,
                n_realisations=n_realisations,
                add_set_name="retrack_sets.leading_edge_weight_factor",
            )
        )

    df = pd.concat(fit_res_list, ignore_index=True)

    assert df.shape[0] == (swhs.size * n_realisations)
    assert "leading_edge_weight_factor" in df.columns


def test_sim_swh_rmse(monte_sim):
    df = monte_sim.multi_proc(swh=np.arange(0.5, 15, 2), n_realisations=1)

    var_types_list = ["swh", "epoch_ns", "Pu"]
    fig, axs = plt.subplots(len(var_types_list), len(default_cost_functions))
    ax_it = iter(axs.ravel())

    for var_type in ["swh", "epoch_ns", "Pu"]:
        for cfunc in default_cost_functions:
            plot_cost_func_vs_swh(
                df, var_type=var_type, cost_func=cfunc, ax=next(ax_it)
            )

    fig.show()


def test_sim_swh_rmse_different_le_weight_factor(monte_sim):
    param_name = "retrack_sets.leading_edge_weight_factor"
    param_name_short = param_name.split(".")[-1]
    params_vals = [1.0, 2.0, 4.0, 8.0]
    add_interference = True
    n_realisations = 1

    # run Monte-Carlo sim with different varying parameters
    # df = pd.DataFrame()
    df_list = []
    for v in params_vals:
        monte_sim.retrack_sets.leading_edge_weight_factor = v
        df_list.append(
            monte_sim.multi_proc(
                swh=np.arange(1, 15, 2),
                n_realisations=n_realisations,
                add_set_name=param_name,
                # max_workers=1,
                # add_interference=add_interference
            )
        )

    df = pd.concat(df_list)

    # prepare plots
    var_types_list = ["swh", "epoch_ns", "Pu"]
    fig, axs = plt.subplots(len(var_types_list), len(default_cost_functions))
    ax_it = iter(axs.ravel())

    # plotting
    for var_type in var_types_list:
        for cfunc in default_cost_functions:
            ax = next(ax_it)
            for v in params_vals:
                sub_df = df.query(f"{param_name_short} == {v}")

                plot_cost_func_vs_swh(
                    sub_df,
                    var_type=var_type,
                    cost_func=cfunc,
                    ax=ax,
                    ax_kwargs={"label": f"{param_name_short}: {v}"},
                )

    fig.suptitle(
        f"Settings-Preset={settings_preset.value.upper()}, N={n_realisations}, state_interference={str(add_interference)}",
        fontsize=6,
    )
    fig.show()


def test_sim_swh_rmse_oversampling():
    param_name = "wf_sets_retracker.internal_oversampling_factor"
    param_name_short = param_name.split(".")[-1]
    params_vals = [1.0, 4.0]
    # params_vals = [4.0]
    # add_thermal_speckle_noise = True
    add_thermal_speckle_noise = False
    add_interference = False
    # n_realisations = 20
    n_realisations = 1
    swh_step = 0.25
    # swh_step = 0.05
    swh_vals = np.arange(-0.125, 20, swh_step)

    # run Monte-Carlo sim with different varying parameters
    df_list = []
    for v in params_vals:
        wf_sets_retracker = WaveformSettings.get_default_src_type(
            L1bSourceType.EUM_S3, internal_oversampling_factor=v
        )
        wf_sets_model = WaveformSettings.get_default_src_type(
            L1bSourceType.EUM_S3, internal_oversampling_factor=1.0
        )
        monte_sim = MonteCarloSimulator(
            retrack_sets=retrack_sets,
            fitting_sets=fitting_sets,
            model_sets=model_sets,
            wf_sets_retracker=wf_sets_retracker,
            wf_sets_model=wf_sets_model,
        )

        df_list.append(
            monte_sim.multi_proc(
                swh=swh_vals,
                n_realisations=n_realisations,
                add_set_name=param_name,
                # max_workers=1,
                add_thermal_speckle_noise=add_thermal_speckle_noise,
                add_interference=add_interference,
            )
        )

    df = pd.concat(df_list)

    # prepare plots
    # var_types_list = ['swh', 'epoch_ns', 'Pu']
    var_types_list = ["swh"]
    fig, axs = plt.subplots(
        len(var_types_list) + 1, len(default_cost_functions), dpi=300
    )
    ax_it = iter(axs.ravel())

    # plotting cost funcs for
    for var_type in var_types_list:
        for cfunc in default_cost_functions:
            ax = next(ax_it)
            for v in params_vals:
                sub_df = df.query(f"{param_name_short} == {v}")

                plot_cost_func_vs_swh(
                    sub_df,
                    var_type=var_type,
                    cost_func=cfunc,
                    ax=ax,
                    ax_kwargs={"label": f"{param_name_short}: {v}"},
                )

    # SWH STD plot
    ax = next(ax_it)
    for v in params_vals:
        sub_df = df.query(f"{param_name_short} == {v}")

        plot_cost_func_vs_swh(
            sub_df,
            var_type="swh",
            cost_func=CostFunctionType.SDD,
            ax=ax,
            ax_kwargs={"label": f"{param_name_short}: {v}"},
        )

    # SWH corrections plot
    swh_ranges = np.arange(-0.5, 20 + swh_step, swh_step)
    df_1 = df.query(f"{param_name_short} == {1.0}")
    df_4 = df.query(f"{param_name_short} == {4.0}")

    ax = next(ax_it)
    df_median_bias_vs_swh_exp_1 = df_1.groupby(
        pd.cut(df_1["swh"], swh_ranges, right=False)
    ).apply(
        lambda g: cost_functions[CostFunctionType.MEDIAN_BIAS](
            g["swh"], g["swh_expected"]
        )
    )
    df_median_bias_vs_swh_exp_4 = df_4.groupby(
        pd.cut(df_4["swh"], swh_ranges, right=False)
    ).apply(
        lambda g: cost_functions[CostFunctionType.MEDIAN_BIAS](
            g["swh"], g["swh_expected"]
        )
    )
    df_diff = df_median_bias_vs_swh_exp_4 - df_median_bias_vs_swh_exp_1
    df_diff *= -1  # the correction shall latter be added to the estimated SWH

    # modulate corr below boundary where there are  no values
    ind_first_valid_val = (~np.isnan(df_diff[0:10])).values.nonzero()[0][0]
    m = df_diff[ind_first_valid_val] / ind_first_valid_val
    df_diff[:ind_first_valid_val] = np.arange(ind_first_valid_val) * m

    fontsize_ticks = 7
    fontsize_xylabel = 6

    # replace ranges with center bins (for plotting)
    df_diff.index = [i.mid for i in df_diff.index]
    df_diff = df_diff.fillna(value=0.0)
    _ax = df_diff.plot(ax=ax, marker="o", markersize=3)

    _ax.grid()
    _ax.set_xticks(np.arange(0, int(df_diff.index.values[-1]) + 1))
    _ax.tick_params(labelsize=fontsize_ticks)
    _ax.set_xlabel("estimated SWH [m]", fontsize=fontsize_xylabel)
    _ax.set_ylabel("SWH correction [m]", fontsize=fontsize_xylabel)

    fig.suptitle(
        f"Settings-Preset={settings_preset.value.upper()}, "
        f"N={n_realisations}, "
        f"state_thermal_speckle_noise={str(add_thermal_speckle_noise)}, "
        f"state_interference={str(add_interference)}",
        fontsize=6,
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.95)
    fig.show()

    # save corr LUT to .pickle
    # dest_pickle_swh_corr = Path.cwd().parent.parent / 'scripts' / f'{settings_preset.value.lower()}_swh_corr_lut.pickle'
    # pickle.dump({'lut_swh_corr': np.asarray(df_diff), 'swh_ranges': np.asarray(swh_ranges)}, open(dest_pickle_swh_corr, 'wb'))
    # swh_ranges_center = swh_ranges[:-1] + swh_step
    # plot_corr_lut_table(swh_ranges_center=swh_ranges_center, corr_lut=np.asarray(df_diff), title=dest_pickle_swh_corr.name)
    # print(f'stored SWH_corr_LUT to {dest_pickle_swh_corr}')


def test_swh_epoch_scatter(monte_sim):
    df = monte_sim.multi_proc(swh=np.arange(2.0), n_realisations=5)

    fig, ax = plt.subplots()
    plot_swh_epoch_scatter(df, ax=ax)
    fig.show()

    print("bla")
