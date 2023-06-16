import logging
from concurrent import futures
from enum import Enum
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pysamosa import retracker
from pysamosa.common_types import (
    SENSOR_SETS_DEFAULT_S3,
    FittingSettings,
    ModelSettings,
    RetrackerSettings,
    WaveformSettings,
)
from pysamosa.data_access import get_model_param_obj_from_l1b_data
from pysamosa.l1b_simulator import L1bSimulator


class CostFunctionType(Enum):
    RMSE = "rmse"
    SDD = "std"
    MEDIAN_BIAS = "median_bias"


default_cost_functions = [
    CostFunctionType.RMSE,
    CostFunctionType.MEDIAN_BIAS,
]


cost_functions = {
    CostFunctionType.RMSE: lambda x, y: np.sqrt(np.mean((x - y) ** 2)),
    CostFunctionType.SDD: lambda x, y: np.std(x - y),
    CostFunctionType.MEDIAN_BIAS: lambda x, y: np.median(x - y),
}


def montesim_call_wrapper(self, kwargs, swh):
    """Wrapper function for ProcessPoolExecutor.map in combination with functools.partial.
    self, kwargs arguments must come first. swh is the one to iterate over.

    :param self: the object of the MonteCarloSimulator
    :param kwargs: the fixed arguments on the call on __call__
    :param swh: the argument to iterate over.
    :return: the wrapped function object (__call__ function of self)
    """
    return self.__call__(swh, **kwargs)


class MonteCarloSimulator:
    def __init__(
        self,
        retrack_sets: RetrackerSettings,
        fitting_sets: FittingSettings,
        model_sets: ModelSettings,
        wf_sets_retracker: WaveformSettings,
        wf_sets_model: WaveformSettings = None,
    ):
        self.retrack_sets = retrack_sets
        self.fitting_sets = fitting_sets
        self.model_sets = model_sets
        self.wf_sets_retracker = wf_sets_retracker
        self.wf_sets_model = (
            wf_sets_model if wf_sets_model is not None else wf_sets_retracker
        )

        self.sr = retracker.SamosaRetracker(
            retrack_sets=retrack_sets,
            fitting_sets=fitting_sets,
            wf_sets=wf_sets_retracker,
            sensor_sets=SENSOR_SETS_DEFAULT_S3,
        )

    def __call__(
        self,
        swh,
        *,
        n_realisations,
        add_set_name=None,
        add_thermal_speckle_noise=True,
        add_interference=False,
    ):
        add_set_name = [] if add_set_name is None else add_set_name
        add_set_name = (
            add_set_name if isinstance(add_set_name, list) else [add_set_name]
        )

        l1b_sim = L1bSimulator(
            model_sets=self.model_sets,
            wf_sets=self.wf_sets_model,
            settings_preset=self.retrack_sets.settings_preset,
            swh=swh,
            Pu=1.0,
            add_thermal_speckle_noise=add_thermal_speckle_noise,
            add_interference=add_interference,
        )
        l1b_sim_it = iter(l1b_sim)

        logging.info(
            f"Running montecarlo simulation with SWH={l1b_sim.swh}, epoch={l1b_sim.epoch_ns}ns, Pu_norm={l1b_sim.Pu}"
        )

        fit_results = []
        for i in range(n_realisations):
            l1b_data_single = next(l1b_sim_it)
            model_params = get_model_param_obj_from_l1b_data(l1b_data_single, 0)

            res_fit = self.sr.fit_wf(
                l1b_data_single=l1b_data_single, model_params=model_params
            )

            # append expected values: swh, epoch_ns, Pu
            res_fit = {
                **res_fit,
                **{
                    "swh_expected": l1b_sim.swh,
                    "epoch_ns_expected": l1b_sim.epoch_ns,
                    "Pu_expected": l1b_sim.Pu,
                },
            }

            # append additional settings columns
            # hack because of NameError when eval tries to evaluate self
            symbols = {"self": self}
            res_fit = {
                **res_fit,
                **{
                    col.split(".")[-1]: eval(f"self.{col}", symbols)
                    for col in add_set_name
                },
            }

            fit_results.append(res_fit)

        return pd.DataFrame(fit_results)

    def multi_proc(self, swh, max_workers=None, **kwargs):
        swh = np.asarray(swh)

        if max_workers == 1:
            fit_results_list = []
            for s in swh:
                fit_results_list.append(self.__call__(s, **kwargs))
        else:
            with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                # treat kwargs as fixed arguments for monte_sim.__call__ and
                # swh as variable param on each process to parallelise
                fit_results_list = pool.map(
                    partial(montesim_call_wrapper, self, kwargs), swh, chunksize=5
                )

        df = pd.concat(fit_results_list, ignore_index=True)

        return df


def plot_cost_func_vs_swh(
    df,
    *,
    var_type,
    cost_func: CostFunctionType,
    ax=None,
    ax_kwargs=None,
    legend_kwargs=None,
    label_kw=None,
    ticks_kw=None,
    subplot_name=None,
    subplot_name_kw=None,
):
    _ax = ax if ax is not None else plt.gca()
    ax_kwargs = ax_kwargs if ax_kwargs is not None else {}
    default_legend_kwargs = {"fontsize": 5}
    legend_kwargs = (
        {**default_legend_kwargs, **legend_kwargs}
        if legend_kwargs is not None
        else default_legend_kwargs
    )

    default_label_kw = {"fontsize": 5}
    label_kw = (
        {**default_label_kw, **label_kw} if label_kw is not None else default_label_kw
    )

    default_ticks_kw = {"labelsize": 5}
    ticks_kw = (
        {**default_ticks_kw, **ticks_kw} if ticks_kw is not None else default_ticks_kw
    )

    subplot_name_kw = subplot_name_kw if subplot_name_kw is not None else {}

    func = cost_functions[cost_func]
    df_func_vs_swh_exp = df.groupby(by="swh_expected").apply(
        lambda g: func(g[var_type], g[f"{var_type}_expected"])
    )

    _ax = df_func_vs_swh_exp.plot(marker="o", markersize=3, ax=_ax, **ax_kwargs)

    label_unit = {
        "swh": ("SWH", "m"),
        "epoch_ns": ("Epoch", "ns"),
        "Pu": ("Pu", "norm"),
    }

    _ax.set_xlabel(
        f"expected {label_unit[var_type][0]} [{label_unit[var_type][1]}]", **label_kw
    )
    _ax.set_ylabel(
        f'{label_unit[var_type][0]} {cost_func.value.lower().replace("_", " ")} [{label_unit[var_type][1]}]',
        **label_kw,
    )
    _ax.set_xticks(np.arange(0, int(df.swh_expected.values[-1]) + 1, 2))
    _ax.tick_params(**ticks_kw)
    _ax.grid()
    _ax.legend(**legend_kwargs)

    if subplot_name:
        _ax.annotate(subplot_name, **subplot_name_kw)

    return _ax


def plot_swh_epoch_scatter(df, ax=None):
    _ax = ax
    # _ax = ax if ax is not None else plt.gca()

    def swh_epoch_diff(g):
        return pd.Series(
            {
                "swh_diff": g["swh"] - g["swh_expected"],
                "epoch_ns_diff": g["epoch_ns"] - g["epoch_ns_expected"],
            }
        )

    new_df = df.apply(swh_epoch_diff, axis=1)
    _ax = new_df.plot.scatter("swh_diff", "epoch_ns_diff", ax=_ax)
    _ax.set_xlabel("SWH RMSE [m]")
    _ax.set_ylabel("Epoch RMSE [m]")
    _ax.grid()

    return _ax
