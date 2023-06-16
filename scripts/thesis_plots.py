import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysamosa import retracker
from pysamosa.common_types import (
    L1bSourceType,
    SensorSettings,
    SensorType,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.data_access import (
    _read_dataset_vars_from_ds,
    data_vars_s6,
    get_model_param_obj_from_l1b_data,
    get_subset_dataset,
)
from pysamosa.model import ModelParameter, ModelSettings, SamosaModel
from pysamosa.settings_manager import get_default_base_settings
from pysamosa.utils import gen_first_true

default_width_in = 5.79  # in inches
default_ratio = 0.5 + np.sqrt(5) / 2  # 1.618, golden ratio
default_figsize_in = [default_width_in, default_width_in / default_ratio]

# set_pgf_mode()

okabeito_colorblind_colors = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddishpurple": "#CC79A7",
    "black": "#000000",
}

THESIS_BASEDIR = Path("/home/schlembach/fastdata/repos/phd-project-flo/thesis/figures/")

fig_list = {}
fig_scale_factor = 2
figwidth_textwidth = fig_scale_factor * default_figsize_in[0]
figheight_textwidth = fig_scale_factor * default_figsize_in[1]
default_figsize = [figwidth_textwidth, figheight_textwidth]
default_dpi = 400
default_subplot_name_kw = {
    "xycoords": "axes fraction",
    "xy": (0.02, 0.96),
    "textcoords": "offset pixels",
    "xytext": (0, 0),
    "horizontalalignment": "left",
    "verticalalignment": "top",
    "fontweight": "bold",
    "color": "black",
    "bbox": dict(
        boxstyle="square,pad=.2",
        facecolor="white",
        edgecolor="none",
        alpha=0.7,
        zorder=10,
    ),
}
default_legend_kw = {"fontsize": 8, "labelspacing": 0.20, "borderaxespad": 0.2}


def gen_multilooked_vary_swh():
    fig, ax = plt.subplots(figsize=[5, 3], dpi=default_dpi)
    fig_list["samosa_model_vary_swh"] = fig

    cols_it = iter(list(okabeito_colorblind_colors.values()))

    settings_preset = SettingsPreset.NONE
    wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.EUM_S6_F04)
    model_sets = ModelSettings()
    sensor_sets = SensorSettings.get_default_sets(st=SensorType.S6_F06)
    sm = SamosaModel(
        model_sets=model_sets,
        wf_sets=wf_sets,
        sensor_sets=sensor_sets,
        settings_preset=settings_preset,
    )

    swh = [0.25, 0.75, 1.0, 2.0, 5, 7, 9]
    Pu = 1.0
    t_0 = -250
    nu = 0

    epoch_ref_gate = 256
    model_params = ModelParameter(epoch_ref_gate=epoch_ref_gate)
    model_params.pri_hz = 1 / 9.2e3
    sensor_sets.bri = 1 / 140
    for s in swh:
        wf = sm.get_waveform_multilook(
            Pu=Pu, Hs=s, t0_ns=t_0, nu=nu, model_params=model_params
        )
        # wf = wf + np.random.randn(len(wf)) * 0.01 * np.nanmax(wf)  # NOISE
        # ADDITION (not in the SAMOSA documentation )
        print(
            f"SWH={s} m, argmax={wf.argmax()}, retracking_point={np.abs(wf - 0.8422).argsort()[1]}"
        )
        ax.plot(wf, label=f"SWH={s:.2f} m", linewidth=1, color=next(cols_it))

    # plot epoch reference gate vertical line
    ax.axvline(
        epoch_ref_gate,
        linestyle="--",
        linewidth="1.0",
        label="Epoch reference gate",
        color=next(cols_it),
    )

    # plot arrow between retracking point and epoch reference gates
    level_retracking_point = 0.8422
    x_retracking_point = np.abs(wf - level_retracking_point).argsort()[0]
    ax.annotate(
        "",
        xy=(x_retracking_point, level_retracking_point),
        xytext=(epoch_ref_gate, level_retracking_point),
        arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, color="black"),
    )
    ax.annotate(
        "epoch",
        xy=(
            x_retracking_point + (epoch_ref_gate - x_retracking_point) / 2,
            level_retracking_point + 0.02,
        ),
        horizontalalignment="center",
    )

    x_pu, y_pu = 25, 1.0
    ax.annotate(
        "",
        xy=(x_pu, 0.0),
        xytext=(x_pu, y_pu),
        arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, color="black"),
    )
    ax.annotate("$P_u$", xy=(x_pu - 5, y_pu / 2), horizontalalignment="right")

    ax.set_xlabel("Range gate [\\#]")
    ax.set_ylabel("Normalised power")
    ax.legend(**default_legend_kw)
    ax.grid()


def plot_single_retrackings():
    fig, axs = plt.subplots(
        1,
        4,
        figsize=[figwidth_textwidth, figwidth_textwidth / 4],
        dpi=default_dpi,
        sharey="row",
    )
    axs = axs.ravel()
    ax_it = iter(axs)
    subplot_name_it = iter(string.ascii_lowercase)

    first_it = gen_first_true()

    fig_list["samosa_retracked_estimates"] = fig

    list_scenarios = [
        # open ocean
        (
            46720,
            Path(
                "/nfs/DGFI145/C/work_flo/coastal_ffsar/orig_data/f06/P4_1B_HR_____/S6A_P4_1B_HR______20210413T182810_20210413T192223_20220514T131950_3253_015_213_106_EUM__REP_NT_F06.SEN6/measurement.nc"
            ),
        ),
        # coastal: peak after leading edge
        (
            8586,
            Path(
                "/nfs/DGFI145/C/work_flo/coastal_ffsar/orig_data/f06/P4_1B_HR_____/S6A_P4_1B_HR______20210502T223038_20210502T232547_20220505T002438_3309_017_196_098_EUM__REP_NT_F06.SEN6/measurement.nc"
            ),
        ),
        # coastal: peak after leading edge
        (
            8581,
            Path(
                "/nfs/DGFI145/C/work_flo/coastal_ffsar/orig_data/f06/P4_1B_HR_____/S6A_P4_1B_HR______20210502T223038_20210502T232547_20220505T002438_3309_017_196_098_EUM__REP_NT_F06.SEN6/measurement.nc"
            ),
        ),
        # coastal: multipeak scenario
        (
            46624,
            Path(
                "/nfs/DGFI145/C/work_flo/coastal_ffsar/orig_data/f06/P4_1B_HR_____/S6A_P4_1B_HR______20210413T182810_20210413T192223_20220514T131950_3253_015_213_106_EUM__REP_NT_F06.SEN6/measurement.nc"
            ),
        ),
    ]
    datavars_l1b = data_vars_s6["l1b"]
    # datavars_l2 = data_vars_eumetsat_s6['l2']

    l1b_src_type, settings_preset = (
        L1bSourceType.EUM_S6_F06,
        SettingsPreset.NONE,
    )

    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(
        settings_preset=settings_preset,
        l1b_src_type=l1b_src_type,
    )

    for ind, file_l1b in list_scenarios:
        ax = next(ax_it)

        grp = "data_20/ku" if "s6" in str(file_l1b).lower() else None
        l1b_data = _read_dataset_vars_from_ds(
            nc_filename=file_l1b,
            data_var_names=datavars_l1b,
            n_offset=ind,
            n_inds=2,
            group=grp,
        )
        l1b = get_subset_dataset(l1b_data, ind_offset=0)
        # l2_data = _read_dataset_vars_from_ds(nc_filename=file_l2, data_var_names=datavars_l2, n_offset=ind, n_inds=2, group=grp)
        # l2 = get_subset_dataset(l2_data, ind_offset=0)

        model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=0)

        sr = retracker.SamosaRetracker(
            retrack_sets=retrack_sets,
            fitting_sets=fitting_sets,
            sensor_sets=sensor_sets,
            wf_sets=wf_sets,
        )

        # start fitting
        res_fit = sr.fit_wf(l1b_data_single=l1b, model_params=model_params)

        ax.plot(
            l1b["wf"] / np.max(l1b["wf"]),
            color=okabeito_colorblind_colors["blue"],
            label="measured waveform",
        )
        ax.plot(
            res_fit["wf_opt"],
            color=okabeito_colorblind_colors["orange"],
            label=f"fitted SAMOSA2 model,\n"
            f'$P_u$={res_fit["Pu"]:.2f}, SWH={res_fit["swh"]:.2f} m,\n'
            f'epoch={res_fit["epoch_ns"]:.2f} ns, misfit={res_fit["misfit"]:.2f}',
        )

        ax.annotate(f"({next(subplot_name_it)})", **default_subplot_name_kw)
        ax.set_xlabel("Range gate [\\#]")
        if next(first_it):
            ax.set_ylabel("Normalised power")
        ax.set_xticks(np.arange(0, 501, 100))
        legend_fontsize = 6
        ax.legend(
            **{**default_legend_kw, **{"fontsize": legend_fontsize}},
            title=f'S6-MF cycle {int(file_l1b.parent.stem.split("_")[13])}, pass {int(file_l1b.parent.stem.split("_")[14])},\n'
            f'lat={np.degrees(l1b["lat_rad"]):.3f}, lon={np.degrees(l1b["lon_rad"]):.3f},\n'
            f'dist-to-coast={l1b["dist2coast"]:.0f} km',
            title_fontsize=legend_fontsize,
        )
        ax.grid()

        # plot_retrack_result(l1b, l2, res_fit, model_params=model_params, wf_sets=wf_sets, model_sets=sr.model_sets, sensor_sets=sensor_sets,
        # retrack_sets=retrack_sets, ax=ax, show_l2_model=False,
        # show_second_halve=l1b_src_type == L1bSourceType.GPOD)

        # fn = ncfile_l1bs.name if ncfile_l1bs is not None else ''
        # fig.subplots_adjust(top=0.65)
        # fig.subplots_adjust(top=0.77)
        # fig.show()


# generate plots
gen_multilooked_vary_swh()
plot_single_retrackings()

for k, v in fig_list.items():
    v.show()

# export figures
export_in_jpg = []
for format in [
    "pgf",
    "pdf",
]:
    for figname, fig in fig_list.items():
        if figname in export_in_jpg:
            _f = "jpg"
        else:
            _f = format

        try:
            fig.savefig(THESIS_BASEDIR / f"{str(figname)}.{_f}", bbox_inches="tight")
            # fig.savefig(fig_export_dir / f'{figname}.{_f}')
        except Exception as e:
            print(f"error on saving {figname}: {e}")
