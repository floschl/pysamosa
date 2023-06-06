import os
import logging
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tests.helpers import consecutive_regions_from_ind_list

from scipy import signal

default_width_in = 5.79  # in inches
default_ratio = 0.5 + np.sqrt(5) / 2  # 1.618, golden ratio
default_figsize_in = [default_width_in, default_width_in / default_ratio]


def gen_first_true():
    yield True
    while True:
        yield False


def plot_single_retrack_result(
    *,
    ax,
    wf_meas,
    res_fit,
    fg_epoch,
    retrack_sets,
    l1b_data_single,
    fontsize_labels=None,
    legend_kwargs=None,
    subplot_name=None,
    subplot_name_kw=None,
    xlabel=None,
    ylabel=None,
    do_plot_second_halve_only=False,
):
    _ax = ax

    fontsize_labels = 8 if fontsize_labels is not None else fontsize_labels
    legend_kwargs = {} if legend_kwargs is None else legend_kwargs
    subplot_name = "" if subplot_name is None else subplot_name
    subplot_name_kw = {} if subplot_name_kw is None else subplot_name_kw

    plot_data = [
        {"name": r"$\mathbf{w}_\mathrm{r}$", "wf": wf_meas},
        {"name": "$\\mathbf{w}_\\mathrm{SAM2}$", "wf": res_fit["wf_opt"]},
        # {'name': '$\mathbf{w}_{\mathrm{SAM2},i}$\n$(\mathrm{SWH}_\mathrm{aim\_mask}=$' + f'{res_fit["swh"]:.2f}m)', 'wf': res_fit['wf_opt']},
    ]

    start_ind = len(plot_data[0]["wf"]) // 2 if do_plot_second_halve_only else 0

    for d in plot_data:
        _ax.plot(d["wf"][start_ind:], linewidth=0.7, label=d["name"])

    # plot initial first-guess epoch
    fg_epoch_style = {
        "linestyle": "--",
        "linewidth": 1,
        "color": "grey",
        "label": "$k_\\mathrm{DFGE}$",
    }
    fg_epoch -= start_ind
    _ax.axvline(fg_epoch, **fg_epoch_style)

    if "interference_inds" in res_fit:
        distorted_regions = [
            (dr - start_ind)
            for dr in consecutive_regions_from_ind_list(res_fit["interference_inds"])
        ]
        show_legend_entry_it = gen_first_true()
        for br in distorted_regions:
            _ax.axvspan(
                br[0],
                br[-1],
                alpha=0.5,
                color="red",
                label="$\\mathbf{k}_\\mathrm{inf}$"
                if next(show_legend_entry_it)
                else "",
            )

    if "interference_mask" in res_fit and not np.allclose(
        res_fit["interference_mask"], 1.0
    ):
        _ax.plot(
            res_fit["interference_mask"][start_ind:],
            linewidth=0.7,
            label="$\\mathbf{w}_\\mathrm{IR}$",
            color="lime",
        )

    # if res_fit['le_inds'] is not None:
    #     le_inds = res_fit['le_inds']
    #     _ax.axvspan(le_inds[0], le_inds[-1], alpha=0.5, color='lightblue')

    if retrack_sets.subwaveform_mode:
        _ax.axvspan(
            res_fit["max_le_gate"] + retrack_sets.subwaveform_n_gates_after_le,
            len(res_fit["wf"]),
            alpha=0.5,
            color="gray",
        )

    _ax.legend(**legend_kwargs)

    # info box
    fontsize_textboxes = 6
    text_str_elems = [
        f"dist2coast: {l1b_data_single['dist2coast']:.0f} km",
    ]
    _ax.text(
        x=0.95,
        y=0.05,
        s="\n".join(text_str_elems),
        ha="right",
        va="center",
        transform=_ax.transAxes,
        fontsize=fontsize_textboxes,
        bbox=dict(pad=0.5, fc="white", alpha=0.8),
    )

    _ax.set_ylim(bottom=-0.1, top=1.1)
    if xlabel:
        _ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    if ylabel:
        _ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    if subplot_name:
        _ax.annotate(subplot_name, **subplot_name_kw)
    _ax.grid()


def save_figs_to_pdf(figs, pdf_filepath):
    """Saves objects to disk using pickle."""
    os.umask(000)
    os.makedirs(os.path.dirname(pdf_filepath), exist_ok=True)

    pp = PdfPages(pdf_filepath)

    figs = figs if isinstance(figs, list) else [figs]

    for f in figs:
        pp.savefig(f, box_inches="tight", pad_inches=0)

    pp.close()

    logging.info("Figure saved to pdf file {}".format(pdf_filepath))


def set_pgf_mode():
    # register pdf extension with FigureCanvasPgf backend
    # https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib/42768093#42768093
    mpl.backend_bases.register_backend("pgf", FigureCanvasPgf)
    mpl.backend_bases.register_backend("pdf", FigureCanvasPgf)

    mpl.rcParams.update(
        {
            "pgf.rcfonts": False,
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.serif": [],  # use latex default serif font
            "font.sans-serif": [],
            "font.monospace": [],
            "figure.figsize": default_figsize_in,
            "figure.constrained_layout.use": True,
            "pgf.preamble": r"\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}",
        }
    )


def scatter_map(
    lat: np.ndarray, lon: np.ndarray, data: np.ndarray, title="", subplots_kw=None
):
    """Scatter plot of a geophysical on a geophysical map."""
    if not subplots_kw:
        subplots_kw = {}

    proj = ccrs.PlateCarree()

    df = pd.DataFrame.from_dict(
        {
            "data": data,
            "lat": lat,
            "lon": lon,
        }
    )

    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), **subplots_kw)

    (~np.isnan(df.data)).values.nonzero()[0]
    # lat_arr = df.iloc[non_nan_mask].lat
    lat_arr = df.lat.values
    lat_start, lat_end = np.around(lat_arr[0], 6), np.around(lat_arr[-1], 6)
    # lon_arr = df.iloc[non_nan_mask].lon
    lon_arr = df.lon.values
    lon_start, lon_end = np.around(lon_arr[0], 6), np.around(lon_arr[-1], 6)
    # lonminmax = [np.floor(np.min(df.iloc[non_nan_mask].lon)), np.ceil(np.max(df.iloc[non_nan_mask].lon))]
    # lonminmax = [-180.0, 180] if lon_start > 180.0 else [np.floor(np.min(lon_arr)), np.ceil(np.max(lon_arr))]
    # lonminmax = [-180.0, 180] if lon_start > 180.0 else [lon_start, lon_end]
    lonminmax = [lon_start, lon_end]
    latminmax = [lat_start, lat_end]

    ax.set_extent([*lonminmax, *latminmax])
    ax.coastlines()
    ax.set_title(title)

    dataminmax = (-0.25, 15)
    n_ticks = 6
    gl = ax.gridlines(
        xlocs=np.round(np.linspace(*lonminmax, n_ticks), 6),
        ylocs=np.round(np.linspace(*latminmax, n_ticks), 6),
        draw_labels=True,
    )
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.LAKES)
    # ax.add_feature(cfeature.BORDERS)
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.RIVERS)

    # create colormap
    cmap = plt.get_cmap("jet")
    # use white color to mark 'bad' values
    # cmap.set_bad(color='k') # currently not working, seems to be bug in
    # pyplot:

    df_vals = df.dropna(subset=["data"])
    df_nans = df[df["data"].isnull()]

    # plot nans
    plt.scatter(df_nans.lon, df_nans.lat, c="black", s=2, transform=proj, zorder=2)

    # plot non-nans
    cb = plt.scatter(
        df_vals.lon,
        df_vals.lat,
        c=df_vals.data,
        s=2,
        vmin=dataminmax[0],
        vmax=dataminmax[1],
        cmap=cmap,
        transform=proj,
        zorder=4,
    )

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(cb, cax=cax)

    return fig, ax


def plot_l2_results_vs_ref(l2, l2_ref, cog_corr=0.0, fig_title=None):
    fontsize_labels, fontsize_legend = 7, 7

    fig, axs_data = plt.subplots(2, 1)

    # lat = np.degrees(l2.latitude)
    lat = l2.latitude

    # SWH
    swh_diff = l2.swh - l2_ref["swh"]
    rmse_swh = float(np.sqrt(np.mean((swh_diff) ** 2)).values)
    median_bias_swh = np.median(swh_diff)
    lh_retrack = axs_data[0].plot(
        lat,
        l2.swh,
        linewidth=1.0,
        label=f"swh_retrack (std={np.nanstd(l2.swh.values):.2f},"
        f"median={np.nanmedian(l2.swh.values):.2f}, bias_eum={median_bias_swh:.4f}m, rmse_eum={rmse_swh:.4f}m)",
        zorder=2,
    )
    axs_data[0].plot(
        lat,
        l2_ref["swh"],
        linewidth=0.8,
        label=f'swh_ref (std={np.nanstd(l2_ref["swh"]):.2f}, median={np.nanmedian(l2_ref["swh"]):.2f})',
    )

    # SWH quality flag
    axs_data[0].plot(
        lat,
        l2.swh_qual,
        color=lh_retrack[0].get_color(),
        linewidth=1.0,
        linestyle="--",
        label="swh_qual",
    )

    # uncorrected SSH
    ssh_uncorr_rt = l2.altitude - l2.range
    ssh_uncorr_ref = l2_ref["alt_m"] - l2_ref["range"] + cog_corr
    ssh_diff = ssh_uncorr_rt - ssh_uncorr_ref

    rmse_ssh = float(np.sqrt(np.mean((ssh_diff) ** 2)).values)
    median_bias_ssh = np.median(ssh_diff)


    def nanstd_detrend(alt, range):
        non_nan_mask = ~(np.isnan(alt) | np.isnan(range))
        return (
            np.nanstd(signal.detrend(alt[non_nan_mask] - range[non_nan_mask]))
            if any(non_nan_mask)
            else np.nan
        )

    axs_data[1].plot(
        lat,
        ssh_uncorr_rt,
        linewidth=1.0,
        label=f"l2 (std_detrend={nanstd_detrend(l2.altitude, l2.range):.2f}m, median_bias={median_bias_ssh:.4f}m, RMSE={rmse_ssh:.4f}m)",
    )
    axs_data[1].plot(
        lat,
        ssh_uncorr_ref,
        linewidth=1.0,
        label=f'l2_ref (std_detrend={nanstd_detrend(l2_ref["alt_m"], l2_ref["range"]):.2f}m)',
    )

    # plot settings
    axs_data[0].legend(fontsize=fontsize_legend, loc='lower left')
    axs_data[0].set_ylabel("SWH [m]", fontsize=fontsize_labels)
    axs_data[0].grid()
    axs_data[0].tick_params(axis='both', which='major', labelsize=fontsize_labels)

    # axs_data[1].set_ylim([np.nanmin(l2.altitude - l2.range), np.nanmax(l2.altitude - l2.range)])
    axs_data[1].legend(fontsize=fontsize_legend, loc='lower left')
    axs_data[1].set_ylabel("uncorrected SSH [m]", fontsize=fontsize_labels)
    axs_data[1].grid()
    axs_data[1].tick_params(axis='both', which='major', labelsize=fontsize_labels)
    axs_data[1].set_ylim(np.min(l2.altitude - l2.range), np.max(l2.altitude - l2.range))


    if fig_title:
        fig.suptitle(fig_title, fontsize=fontsize_labels)

    return fig, rmse_swh, rmse_ssh
