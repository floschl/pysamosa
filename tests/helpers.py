import numbers
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from pysamosa.common_types import (
    SENSOR_SETS_DEFAULT_S3,
    L1bSourceType,
    ModelSettings,
    RetrackerProcessorSettings,
    RetrackerSettings,
    SensorSettings,
    WaveformSettings,
)
from pysamosa.conf_params import CONST_C
from pysamosa.data_access import get_model_param_obj_from_l1b_data, get_subset_dataset
from pysamosa.model import SamosaModel
from pysamosa.retracker import SamosaRetracker, calc_misfit
from pysamosa.retracker_helpers import get_dynamic_first_guess_epochs
from pysamosa.rip import RipAnalyser, RIPParameters
from pysamosa.utils import consecutive_regions_from_ind_list


def plot_retrack_result(
    l1b_data,
    l2_data,
    res_fit,
    model_params,
    *,
    l1b_srctype=L1bSourceType,
    sensor_sets: SensorSettings = None,
    wf_sets: WaveformSettings = None,
    model_sets: ModelSettings = None,
    retrack_sets: RetrackerSettings = None,
    ax=None,
    title=None,
    legend_kwargs=None,
    legend_wrap_len=None,
    show_l2_model=True,
    show_second_halve=False,
    show_leading_edge=False,
):
    if ax:
        _ax = ax
    else:
        _ax = plt.gca()

    fg_epoch = (
        l1b_data["dynamic_fg_epoch"]
        if "dynamic_fg_epoch" in l1b_data
        else np.argmax(l1b_data["wf"])
    )
    if wf_sets is not None:
        fg_epoch *= wf_sets.internal_oversampling_factor

    wf_len = len(res_fit["wf"])
    wf_sets = (
        wf_sets
        if wf_sets is not None
        else WaveformSettings.get_default_src_type(l1b_srctype)
    )

    if retrack_sets and retrack_sets.normalise_wf_by_fg_region:
        wf_meas = res_fit["wf"] / SamosaRetracker.get_wf_max(
            res_fit["wf"], retrack_sets=retrack_sets, fg_epoch=fg_epoch
        )
    else:
        wf_meas = res_fit["wf"] / np.max(res_fit["wf"])

    sm = SamosaModel(
        sensor_sets=sensor_sets,
        model_sets=model_sets,
        wf_sets=wf_sets,
        settings_preset=retrack_sets.settings_preset,
    )

    Pu_W_l2 = l2_data["Pu_W"] if "Pu_W" in l2_data else np.nan
    Pu_W_fit = res_fit["Pu_denorm"]
    l2_epoch_ns = (
        l2_data["epoch_ns"]
        if "epoch_ns" in l2_data and not np.isnan(l2_data["epoch_ns"])
        else np.nan
    )
    try:
        wf_l2_model = sm.get_waveform_multilook(
            Pu=Pu_W_l2, Hs=l2_data["swh"], t0_ns=l2_epoch_ns, model_params=model_params
        )
    except Exception:
        wf_l2_model = np.zeros(wf_len)
    wf_l2_model = wf_l2_model / np.nanmax(wf_l2_model)
    if show_l2_model:
        wf_l2_model_l2norm = np.sum((wf_meas - wf_l2_model) ** 2)
        misfit_l2_model = calc_misfit(wf_meas, wf_l2_model)
    # ratio_Pu_l2_to_fit = Pu_W_l2 / Pu_W_fit
    # ratio_Pu_l2_to_wf = Pu_W_l2 / np.max(l1b_data['wf'])

    # data to be plotted
    range_l2 = l2_data["range"]
    tracker_range_m = (
        l2_data["tracker_range_m"]
        if "tracker_range_m" in l2_data
        else l1b_data["tracker_range_m"]
    )
    range_fit = tracker_range_m + (float(res_fit["epoch_ns"])) * 1e-9 * (CONST_C / 2)
    ssh_unc_l2 = l2_data["alt_m"] - range_l2
    ssh_unc_fit = l1b_data["alt_m"] - range_fit
    n_iter = l2_data["n_iter"] if "n_iter" in l2_data else np.nan
    plot_data = [
        {
            "name": "y_l2",
            "wf": wf_meas,
            "misfit": l2_data.get("misfit", np.nan),
            "l2norm": np.nan,
            "n_iter": n_iter,
            "swh": l2_data["swh"],
            "epoch_ns": float(l2_epoch_ns),
            "range": range_l2,
            "ssh_unc": ssh_unc_l2,
            "Pu_W": Pu_W_l2,
            "nu": np.nan,
        },
        {
            "name": "y_retrack",
            "wf": res_fit["wf_opt"],
            "misfit": res_fit["misfit"],
            "misfit_selective": res_fit["misfit_selective"],
            "misfit_le": res_fit["misfit_le"] if "misfit_le" in res_fit else np.nan,
            "l2norm": res_fit["l2norm"],
            "n_iter": res_fit["n_iter"],
            "swh": res_fit["swh"],
            "epoch_ns": res_fit["epoch_ns"],
            "range": range_fit,
            "ssh_unc": ssh_unc_fit,
            "Pu_W": Pu_W_fit,
            "nu": res_fit["nu"],
        },
    ]

    if show_l2_model:
        plot_data.append(
            {
                "name": "y_meas_model",
                "wf": wf_l2_model,
                "misfit": misfit_l2_model,
                "l2norm": wf_l2_model_l2norm,
                "n_iter": np.nan,
                "swh": l2_data["swh"],
                "epoch_ns": l2_epoch_ns,
                "range": range_l2,
                "ssh_unc": ssh_unc_l2,
                "Pu_W": Pu_W_l2,
                "nu": np.nan,
            }
        )

    legend_len = 90 if legend_wrap_len is None else legend_wrap_len
    for d in plot_data:
        _ax.plot(
            d["wf"],
            linewidth=0.7,
            label="\n".join(
                wrap(
                    f"{d['name']}, misfit={d['misfit']:.2f},"
                    f" misfit_selective={d['misfit_selective'] if 'misfit_selective' in d else np.nan:.2f}, "
                    f" misfit={d['misfit'] if 'misfit' in d else np.nan:.2f}, "
                    # f" misfit_le={(d['misfit_le'] if 'misfit_le' in d else np.nan):.2f}"
                    f" n_iter={d['n_iter']}, "
                    f"SWH={d['swh']:.3f}m, "
                    f"epoch_ns={d['epoch_ns']:.3f}ns,"
                    f"range={d['range']:.3f}m,"
                    f"ssh_unc={d['ssh_unc']:.3f}m,"  # noqa
                    # f"Pu={d['Pu_W']:.2e}W"
                    # f"nu={d['nu']:.2e}"
                    ,
                    width=legend_len,
                )
            ),
        )

    # plot initial first-guess epoch
    fg_epoch_style = {
        "linestyle": "--",
        "linewidth": 1,
        "color": "grey",
        "label": "Dynamic First-Guess Epoch (DFGE)",
    }
    _ax.axvline(fg_epoch, **fg_epoch_style)

    if "interference_inds" in res_fit:
        distorted_regions = consecutive_regions_from_ind_list(
            res_fit["interference_inds"]
        )
        for br in distorted_regions:
            _ax.axvspan(br[0], br[-1], alpha=0.5, color="red")

    if "interference_mask" in res_fit and not np.allclose(
        res_fit["interference_mask"], 1.0
    ):
        _ax.plot(
            res_fit["interference_mask"],
            linewidth=0.7,
            label="interference reference waveform",
        )

    if show_leading_edge and res_fit["le_inds"] is not None:
        le_inds = res_fit["le_inds"]
        _ax.axvspan(
            le_inds[0], le_inds[-1], alpha=0.5, color="lightblue", label="leading edge"
        )

    if retrack_sets is not None and retrack_sets.subwaveform_mode:
        _ax.axvspan(
            res_fit["max_le_gate"] + retrack_sets.subwaveform_n_gates_after_le,
            wf_len,
            alpha=0.5,
            color="gray",
        )

    # labelling
    fontsize_labels = 7
    fontsize_textbox = 6
    _ax.set_title("" if title is None else title, fontsize=fontsize_labels)
    legend_kwargs_default = {
        "loc": "lower left",
        "bbox_to_anchor": (0, 1.03, 1, 0),
        "mode": "expand",
        **{
            "prop": {"size": fontsize_textbox},
            "labelspacing": 0.50,
            "borderaxespad": 0.2,
        },
    }
    if legend_kwargs is not None:
        legend_kwargs_default = {**legend_kwargs_default, **legend_kwargs}
    _ax.legend(**legend_kwargs_default)

    # info box
    fontsize_textboxes = 6
    text_str_elems = [
        f"dist2coast: {l1b_data['dist2coast']:.0f} km",
    ]
    _ax.text(
        x=0.95,
        y=0.95,
        s="\n".join(text_str_elems),
        ha="right",
        va="center",
        transform=_ax.transAxes,
        fontsize=fontsize_textboxes,
        bbox=dict(pad=0.5, fc="white", alpha=0.8),
    )

    if show_second_halve:
        _ax.set_xlim(left=wf_len // 2, right=wf_len)
    _ax.set_ylim(bottom=-0.1, top=1.1)
    _ax.set_xlabel("Range gate k", fontsize=fontsize_labels)
    _ax.set_ylabel("Normalised power", fontsize=fontsize_labels)
    _ax.grid()

    if not ax:
        plt.show()


def plot_rip_result(
    l1b_data_single: dict, ripa: RipAnalyser, rip_params: RIPParameters, ax=None
):
    if ax is not None:
        _ax = ax
    else:
        _ax = plt.gca()

    _ax.plot(ripa.doppler_beam_inds, ripa.rip_wf_norm, label="measured RIP")
    _ax.plot(
        ripa.doppler_beam_inds, rip_params.rip_az_fitted, "+", label="fitted gaussian"
    )
    _ax.axvline(
        rip_params.pitch_mispoint_look, color="red", linestyle="dashed", linewidth=0.5
    )
    _ax.legend(fontsize=7, loc="upper left")

    _ax.plot(
        rip_params.halfpower_looks,
        0.5 * rip_params.amplitude_fitted_norm * np.ones(2),
        color="red",
        linestyle="dashed",
        linewidth=0.5,
        label="3db aperture",
    )

    # labelling
    fontsize = 7
    fontsize_textboxes = 6

    # RIP parameter text box
    text_str = "\n".join(
        [
            f"{k} = {v:.3e}"
            for k, v in rip_params.dict().items()
            if isinstance(v, numbers.Number)
        ]
    )
    _ax.text(
        x=0.98,
        y=0.05,
        s=text_str,
        ha="right",
        va="bottom",
        transform=_ax.transAxes,
        fontsize=fontsize_textboxes,
        bbox=dict(pad=0.4, fc="white", alpha=0.8),
    )

    # measured params
    l1b_params = {"pitch_mispoint_rad_instr": l1b_data_single["ksix_rad"]}
    text_str_elems = [*[f"{k} = {v:.3e}" for k, v in l1b_params.items()]]
    _ax.text(
        x=0.98,
        y=0.95,
        s="\n".join(text_str_elems),
        ha="right",
        va="top",
        transform=_ax.transAxes,
        fontsize=fontsize_textboxes,
        bbox=dict(pad=0.4, fc="white", alpha=0.8),
    )

    _ax.set_xlabel("integer looks", fontsize=fontsize)
    _ax.set_ylabel("normalised power", fontsize=fontsize)

    if not ax:
        plt.show()


def get_l1b_data_single_from_l1b_fixture(
    *, l1b_fixture, ind, rp_sets: RetrackerProcessorSettings, file_id
):
    n_inds_total = rp_sets.dynamic_fg_epoch_n_adjacent_meas
    n_pre = n_inds_total // 2

    l1b_data = l1b_fixture(n_offset=ind - n_pre, n_inds=n_inds_total, file_id=file_id)
    nc_file = l1b_fixture.get_nc_filename(file_id)
    l1b_data["dynamic_fg_epoch"] = get_dynamic_first_guess_epochs(
        wfs=l1b_data["wf"],
        tracker_range=l1b_data["tracker_range_m"],
        alt_m=l1b_data["alt_m"],
        bu_bw_Hz=SENSOR_SETS_DEFAULT_S3.B_r_Hz,
        fg_epoch_adjacent_meas=rp_sets.dynamic_fg_epoch_n_adjacent_meas,
    )
    l1b_data_single = get_subset_dataset(l1b_data, ind_offset=n_pre)
    model_params_single = get_model_param_obj_from_l1b_data(l1b_data, ind=n_pre)

    return l1b_data_single, model_params_single, nc_file
