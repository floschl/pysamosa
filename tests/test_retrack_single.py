import logging
import tempfile
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from pysamosa import retracker, simple_logger
from pysamosa.common_types import L1bSourceType, ModelSettings, ProcMode, SettingsPreset
from pysamosa.data_access import (
    data_vars_dart,
    data_vars_retracker,
    data_vars_s6,
    get_model_param_obj_from_l1b_data,
    get_subset_dataset,
)
from pysamosa.retracker_helpers import get_dynamic_first_guess_epochs
from pysamosa.settings_manager import get_default_base_settings
from tests.helpers import plot_retrack_result
from tests.settings_dumper import SettingsDumper

rel_inds, file_id = (
    # list(range(46403, 46403+60)), 's3_0',  # nice coastal retracking
    # scenario from CORAL paper
    [25480],
    "s3_0",  # nice coastal retracking scenario from CORAL paper
    # [1000], 'cs_0',
    # [3503], 'cs_1',
    # [3503], 'cs_2',
    # [13344], 's6_eum_0_f03',
    # [18600], 's6_eum_1_f04',
    # [18600], 's6_eum_2_f06',
    #
    # [7], 'dart_s6_0',
    # list(range(55,60)), 'dart_s6_0',
)

conf = [
    (L1bSourceType.EUM_S3, SettingsPreset.NONE),
    # (L1bSourceType.EUM_S3, SettingsPreset.CORALv1),
    ##
    # CS
    # (L1bSourceType.EUM_CS, SettingsPreset.NONE),
    # (L1bSourceType.EUM_CS, SettingsPreset.SAMPLUS),
    # S6
    # (L1bSourceType.EUM_S6_F04, SettingsPreset.NONE),
    # (L1bSourceType.EUM_S6_F04, SettingsPreset.NONE),
    # (L1bSourceType.EUM_S6_F06, SettingsPreset.NONE),
    # (L1bSourceType.EUM_S6_F04, SettingsPreset.SAMPLUS),
    # (L1bSourceType.EUM_S6_F06, SettingsPreset.CORALv2),
    # S6 FFSAR
    # (L1bSourceType.EUM_S6_FFSAR, SettingsPreset.NONE),
    # (L1bSourceType.EUM_S6_FFSAR, SettingsPreset.CORALv2),
]
l1b_src_type, settings_preset = conf[0]

rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(
    settings_preset=settings_preset,
    l1b_src_type=l1b_src_type,
)
model_sets = ModelSettings.get_default_sets(
    st=sensor_sets.sensor_type,
    wf_sets=wf_sets,
)


custom_proc_mode = None
# custom_proc_mode = ProcMode.FFSAR
# custom_proc_mode = ProcMode.UFSAR

custom_file_l1b, custom_file_l2 = None, None
if custom_proc_mode is not None:
    custom_file_l1b = Path(
        "/nfs/DGFI145/C/work_flo/coastal_ffsar/orig_data/f06/P4_1B_HR_____/S6A_P4_1B_HR______20210502T223038_20210502T232547_20220505T002438_3309_017_196_098_EUM__REP_NT_F06.SEN6/measurement.nc"
    )
    custom_file_l2 = Path(
        "/nfs/DGFI145/C/work_flo/coastal_ffsar/orig_data/f06/P4_2__HR_____/S6A_P4_2__HR______20210413T182810_20210413T192223_20220514T174711_3253_015_213_106_EUM__REP_NT_F06.SEN6/S6A_P4_2__HR_STD__NT_015_213_20210413T182810_20210413T192223_F06.nc"
    )
    rel_inds, file_id = list(range(8580, 8595)), None

    if "_DAR_" in str(custom_file_l1b):
        custom_datavars_l1b = (
            data_vars_dart["l1b"]
            if custom_proc_mode is ProcMode.UFSAR
            else data_vars_dart["l1b_ffsar"]
        )
    else:
        custom_datavars_l1b = data_vars_s6["l1b"]

    with xr.open_dataset(custom_file_l2, decode_times=False) as ds:
        if "processor" in ds.attrs and "DGFI" in ds.processor:
            custom_datavars_l2 = data_vars_retracker["l2"]
        else:
            custom_datavars_l2 = data_vars_s6["l2"]

    retrack_sets.fit_zero_doppler = (
        custom_proc_mode and custom_proc_mode is ProcMode.FFSAR
    )


@pytest.fixture(scope="class")
def set_dumper():
    tempdir = Path(tempfile.gettempdir())
    dest_mat_file = (
        tempdir
        / f"retrack_log_{file_id}_inds{rel_inds[0]}-{rel_inds[-1]}_{l1b_src_type.value}_{settings_preset.value}.mat"
    )
    dest_gif_file = (
        tempdir
        / f"retrack_log_{file_id}_inds{rel_inds[0]}-{rel_inds[-1]}_{l1b_src_type.value}_{settings_preset.value}.gif"
    )
    settings_dump = SettingsDumper(
        wf_sets=wf_sets,
        model_sets=model_sets,
        sensor_sets=sensor_sets,
        dest_mat_file=dest_mat_file,
        dest_gif_file=dest_gif_file,
    )
    yield settings_dump
    settings_dump.write_out_mat()
    settings_dump.export_gif()


fig_list = []


class TestRetrackingSingle:
    @pytest.mark.parametrize("rel_ind", rel_inds)
    def test_retracking_single(
        self,
        set_dumper,
        rel_ind,
        s3_eum_l1b,
        s3_eum_l2,
        cs_eum_l1b,
        cs_eum_l2,
        s6_eum_l1b,
        s6_eum_l2,
        s6_dart_l1b,
        s6_dart_l2,
        dataset_generic_l1b,
        dataset_generic_l2,
    ):
        n_offset_l2 = None

        if custom_proc_mode:
            l1b, l2 = dataset_generic_l1b(
                custom_file_l1b, custom_datavars_l1b
            ), dataset_generic_l2(custom_file_l2, custom_datavars_l2)
        elif l1b_src_type == L1bSourceType.EUM_S3:
            l1b, l2 = s3_eum_l1b, s3_eum_l2
        elif l1b_src_type == L1bSourceType.EUM_CS:
            l1b, l2 = cs_eum_l1b, cs_eum_l2
        elif (
            l1b_src_type == L1bSourceType.EUM_S6_F04
            or l1b_src_type == L1bSourceType.EUM_S6_F06
        ):
            l1b, l2 = s6_eum_l1b, s6_eum_l2
        elif (
            l1b_src_type == L1bSourceType.EUM_S6_F04_FFSAR
            or l1b_src_type == L1bSourceType.EUM_S6_F06_FFSAR
        ):
            l1b, l2 = s6_dart_l1b, s6_dart_l2
            retrack_sets.fit_zero_doppler = True

        ncfile_l1bs = l1b.get_nc_filename(file_id)
        if set_dumper and set_dumper.l1b_nc_filename is None:
            set_dumper.l1b_nc_filename = ncfile_l1bs

        n_offset = rel_ind

        # analyse and plot data
        fig, ax = plt.subplots()

        # calculate dynamic firt-guess epoch for a single measurement
        if rp_sets.do_dynamic_fg_epoch:
            n_inds_total = rp_sets.dynamic_fg_epoch_n_adjacent_meas
            n_before = n_inds_total // 2

            n_offset_fg = 0 if (n_offset - n_before) < 0 else n_offset - n_before
            n_target_ind_l1b = n_offset if n_offset_fg == 0 else n_before
            l1b_data = l1b(n_offset=n_offset_fg, n_inds=n_inds_total, file_id=file_id)
            l1b_data["dynamic_fg_epoch"] = get_dynamic_first_guess_epochs(
                wfs=l1b_data["wf"],
                tracker_range=l1b_data["tracker_range_m"],
                alt_m=l1b_data["alt_m"],
                bu_bw_Hz=sensor_sets.B_r_Hz,
                fg_epoch_adjacent_meas=rp_sets.dynamic_fg_epoch_n_adjacent_meas,
            )
            l1b_data_single = get_subset_dataset(l1b_data, ind_offset=n_target_ind_l1b)
        else:
            n_target_ind_l1b = 0
            # choose two to get right direction of track
            l1b_data = l1b(n_offset=n_offset, n_inds=2, file_id=file_id)
            l1b_data_single = get_subset_dataset(l1b_data, ind_offset=n_target_ind_l1b)

        try:
            l2_record_inds_all = l2(n_offset=0, n_inds=0, file_id=file_id)[
                "record_inds"
            ]
            n_ind_l2 = np.argwhere(
                l1b_data_single["record_inds"] == l2_record_inds_all
            )[0][0]
        except BaseException:
            n_ind_l2 = n_offset_l2 if (n_offset_l2 is None) else n_offset

        l2_data = l2(n_offset=n_ind_l2, n_inds=1, file_id=file_id)
        l2_data_single = get_subset_dataset(l2_data, ind_offset=0)
        model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=n_target_ind_l1b)

        # simple_logger.set_root_logger()
        simple_logger.set_root_logger(log_level=logging.DEBUG)

        sr = retracker.SamosaRetracker(
            retrack_sets=retrack_sets,
            fitting_sets=fitting_sets,
            sensor_sets=sensor_sets,
            wf_sets=wf_sets,
        )

        # start fitting
        res_fit = sr.fit_wf(l1b_data_single=l1b_data_single, model_params=model_params)

        # plot fitted curve
        plot_retrack_result(
            l1b_data_single,
            l2_data_single,
            res_fit,
            model_params=model_params,
            wf_sets=wf_sets,
            model_sets=sr.model_sets,
            sensor_sets=sensor_sets,
            retrack_sets=retrack_sets,
            ax=ax,
            show_l2_model=False,
            show_second_halve=l1b_src_type == L1bSourceType.GPOD,
        )

        fn = ncfile_l1bs.name if ncfile_l1bs is not None else ""
        fig.suptitle(
            "\n".join(
                wrap(
                    f"{fn}, {settings_preset.value} ({l1b_src_type.value}), record#: {n_ind_l2}",
                    width=70,
                )
            ),
            fontsize=8,
        )
        # fig.subplots_adjust(top=0.65)
        fig.subplots_adjust(top=0.77)
        fig.show()

        if set_dumper:
            set_dumper.add_retrack_entry(
                l1b_data_single=l1b_data_single,
                model_params=model_params,
                record_ind=n_ind_l2,
                res_fit=res_fit,
                l2_data_single=l2_data_single,
            )
            set_dumper.add_fig(fig)
