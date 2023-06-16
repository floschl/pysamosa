import logging
from pathlib import Path

import numpy as np

from pysamosa.common_types import L1bSourceType, ProcMode, SettingsPreset
from pysamosa.data_access import (
    data_vars_cs,
    data_vars_dart,
    data_vars_retracker,
    data_vars_s3,
    data_vars_s6,
)
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings import S6_DATA_DIR
from pysamosa.settings_manager import get_default_base_settings
from pysamosa.utils import plot_l2_results_vs_ref


def raise_wrong_sensor_type(st, st_correct):
    if st.value is not st_correct.value:
        raise RuntimeError("choose correct SensorType")


max_rmse_swh_m = 0.05
max_rmse_ssh_m = 0.01


def test_retrack_s3(s3_eum_l1b, s3_eum_l2):
    l1b, l2 = s3_eum_l1b, s3_eum_l2
    data_vars_l1b = data_vars_s3["l1b"]
    file_id = "s3_0"
    n_offset = 25480

    l1b_src_type, preset = (
        L1bSourceType.EUM_S3,
        SettingsPreset.NONE
        # L1bSourceType.EUM, SettingsPreset.CORALv1
    )

    # generate default settings
    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(
        settings_preset=preset,
        l1b_src_type=l1b_src_type,
    )

    rp_sets.n_offset = n_offset
    rp_sets.n_inds = 60
    rp_sets.n_procs = 1
    rp_sets.skip_if_exists = False

    l1b_file = l1b.get_nc_filename(file_id)

    rp = RetrackerProcessor(
        l1b_source=[l1b_file],
        l1b_data_vars=data_vars_l1b,
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        wf_sets=wf_sets,
        sensor_sets=sensor_sets,
        log_level=logging.DEBUG,
    )
    rp.process()

    # ref datasets
    l2_ref = l2(n_offset=n_offset, n_inds=rp_sets.n_inds, file_id=file_id)

    fig, rmse_swh, rmse_ssh = plot_l2_results_vs_ref(
        rp.output_l2,
        l2_ref,
        cog_corr=0.55590,
        fig_title=f"{preset} ({l1b_src_type.value})",
    )

    assert rmse_swh < max_rmse_swh_m
    assert rmse_ssh < max_rmse_ssh_m


def test_retrack_s6(s6_eum_l1b, s6_eum_l2):
    l1b, l2 = s6_eum_l1b, s6_eum_l2
    data_vars_l1b = data_vars_s6["l1b"]
    # file_id = 's6_eum_0_f03'; n_offset = 13340;
    # file_id = 's6_eum_1_f04'; n_offset = 18600;
    # file_id = 's6_eum_1_f04'; n_offset = 34700;
    file_id = "s6_eum_2_f06"
    n_offset = 18600
    # file_id = 's6_eum_2_f06'; n_offset = 34700;

    l1b_file = l1b.get_nc_filename(file_id)

    l1b_src_type = (
        L1bSourceType.EUM_S6_F04
        if "f04" in str(l1b_file).lower()
        else L1bSourceType.EUM_S6_F06
    )
    preset = SettingsPreset.NONE

    # generate default settings
    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(
        settings_preset=preset,
        l1b_src_type=l1b_src_type,
    )

    rp_sets.n_offset = n_offset
    rp_sets.n_inds = 26
    rp_sets.n_procs = 1
    rp_sets.skip_if_exists = False

    rp = RetrackerProcessor(
        l1b_source=[l1b_file],
        l1b_data_vars=data_vars_l1b,
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        sensor_sets=sensor_sets,
        wf_sets=wf_sets,
        log_level=logging.DEBUG,
    )

    # if 'f04' in str(l1b_file).lower():
    #     rp.retracker.model_sets.Enable_Slope_Effect_Flag = True
    # rp.retracker.model_sets.Enable_Slope_Effect_Flag = True

    rp.process()

    # ref datasets
    l2_ref = l2(n_offset=n_offset, n_inds=rp_sets.n_inds, file_id=file_id)

    fig, rmse_swh, rmse_ssh = plot_l2_results_vs_ref(
        rp.output_l2,
        l2_ref,
        fig_title=f"{preset} ({l1b_src_type.value})",
    )

    fig.show()

    assert rmse_swh < max_rmse_swh_m
    assert rmse_ssh < max_rmse_ssh_m


def test_retrack_cs(cs_eum_l1b, cs_eum_l2):
    l1b_src_type, preset = (
        L1bSourceType.EUM_CS,
        SettingsPreset.NONE,
    )

    l1b, l2 = cs_eum_l1b, cs_eum_l2
    data_vars_l1b = data_vars_cs["l1b"]
    # file_id = 'cs_0'; n_offset = 1000;
    file_id = "cs_1"
    n_offset = 3503
    # file_id = 'cs_2'; n_offset = 3503;

    # generate default settings
    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(
        settings_preset=preset,
        l1b_src_type=l1b_src_type,
    )

    rp_sets.n_offset = n_offset
    rp_sets.n_inds = 30
    rp_sets.n_procs = 1
    rp_sets.skip_if_exists = False

    l1b_file = l1b.get_nc_filename(file_id)

    rp = RetrackerProcessor(
        l1b_source=[l1b_file],
        l1b_data_vars=data_vars_l1b,
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        sensor_sets=sensor_sets,
        wf_sets=wf_sets,
        log_level=logging.DEBUG,
    )
    rp.process()

    # ref datasets
    l2_ref = l2(n_offset=n_offset, n_inds=rp_sets.n_inds, file_id=file_id)

    fig, rmse_swh, rmse_ssh = plot_l2_results_vs_ref(
        rp.output_l2,
        l2_ref,
        fig_title=f"{preset} ({l1b_src_type.value})",
    )

    fig.show()

    # assert rmse_swh < max_rmse_swh_m
    # assert rmse_ssh < max_rmse_ssh_m


def test_retrack_ffsar(dataset_generic_l1b, dataset_generic_l2):
    proc_mode = ProcMode.FFSAR
    # proc_mode = ProcMode.UFSAR

    is_ffsar = proc_mode == ProcMode.FFSAR
    n_offset = 0
    n_inds = 20

    data_vars_l1b = (
        data_vars_dart["l1b_ffsar"]
        if proc_mode == ProcMode.FFSAR
        else data_vars_dart["l1b"]
    )
    data_vars_l2 = data_vars_retracker["l2"]
    ncfile_l1b = Path(
        S6_DATA_DIR
        / "ffsar"
        / "l1b"
        / "S6A_P4_1B_HR______20210423T162639_20210423T172054_20220514T143739_3255_016_213_106_DAR__REP_NT_F06_53.73_53.95.nc"
    )
    l1b = dataset_generic_l1b(ncfile_l1b, data_vars_l1b)

    ncfile_l2 = Path(
        S6_DATA_DIR
        / "ffsar"
        / "l2"
        / "S6A_P4_1B_HR______20210423T162639_20210423T172054_20220514T143739_3255_016_213_106_DAR__REP_NT_F06_53.73_53.95.nc"
    )
    l2 = dataset_generic_l2(ncfile_l2, data_vars_l2)

    preset = SettingsPreset.CORALv2
    # preset = SettingsPreset.NONE

    # generate default settings
    if is_ffsar:
        l1b_src_type = (
            L1bSourceType.EUM_S6_F04_FFSAR
            if "f04" in str(ncfile_l1b).lower()
            else L1bSourceType.EUM_S6_F06_FFSAR
        )
    else:
        l1b_src_type = (
            L1bSourceType.EUM_S6_F06
            if "f06" in str(ncfile_l1b).lower()
            else L1bSourceType.EUM_S6_F04
        )

    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(
        settings_preset=preset,
        l1b_src_type=l1b_src_type,
    )
    wf_sets.np, wf_sets.zp_oversampling_factor = (2 * 256, 2)

    rp_sets.n_offset = n_offset
    rp_sets.n_inds = n_inds
    rp_sets.n_procs = 10
    rp_sets.skip_if_exists = False
    l2_posting_rate = 20
    l1b_posting_rate = (
        int(ncfile_l1b.parent.name.split("_")[-1])
        if "coastal_ffsar" in str(ncfile_l1b)
        else 20
    )
    # rp_sets.reduce_l2_factor = 1
    rp_sets.reduce_l2_factor = round(l1b_posting_rate / l2_posting_rate)
    # rp_sets.reduce_l2_factor = 4 if '_80/' in str(ncfile_l1b) else 1
    rp_sets.dynamic_fg_epoch_n_adjacent_meas *= rp_sets.reduce_l2_factor

    l1b_file = l1b.get_nc_filename()

    rp = RetrackerProcessor(
        l1b_source=[l1b_file],
        l1b_data_vars=data_vars_l1b,
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        sensor_sets=sensor_sets,
        wf_sets=wf_sets,
        # log_level=logging.DEBUG,
    )
    rp.process()

    # ref datasets
    l2_ref = l2(n_offset=0, n_inds=0)
    l2_mask = np.in1d(l2_ref["record_inds"], rp.output_l2.record_ind.values)
    l2_ref = l2(n_offset=l2_mask.nonzero()[0][0], n_inds=len(l2_mask.nonzero()[0]))

    fig, rmse_swh, rmse_ssh = plot_l2_results_vs_ref(
        rp.output_l2,
        l2_ref,
        fig_title=f"{preset} ({l1b_src_type.value})",
    )

    fig.show()

    # assert rmse_swh < max_rmse_swh_m
    # assert rmse_ssh < max_rmse_ssh_m
