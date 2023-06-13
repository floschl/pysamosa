import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pysamosa.dist2coast import get_dist_pacioos

from .common_types import ModelParameter
from .conf_params import CONST_C

data_vars_s3 = {
    "l1b": {
        "wf": "i2q2_meas_ku_l1b_echo_sar_ku",
        "time": "time_l1b_echo_sar_ku",
        "lat_rad": "lat_l1b_echo_sar_ku",
        "lon_rad": "lon_l1b_echo_sar_ku",
        "alt_m": "alt_l1b_echo_sar_ku",
        "h_rate_m_per_s": "orb_alt_rate_l1b_echo_sar_ku",
        "Vs_m_per_s": "x_vel_l1b_echo_sar_ku",
        # 'ksix_rad': 'pitch_sral_mispointing_l1b_echo_sar_ku',  # pitch
        # 'ksiy_rad': 'roll_sral_mispointing_l1b_echo_sar_ku',  # roll
        "beam_ang_stack_rad": "beam_ang_stack_l1b_echo_sar_ku",
        "epoch_ref_gate": "epoch_ref_gate_const",
        "tracker_range_m": "range_ku_l1b_echo_sar_ku",
    },
    "l2": {
        "time": "time_l1bs_echo_sar_ku",
        "lat_rad": "lat_l1bs_echo_sar_ku",
        "lon_rad": "lon_l1bs_echo_sar_ku",
        "swh": "swh_ocean_20_ku",
        "swh_qual": "swh_ocean_qual_20_ku",
        "Pu_W": "amplitude_ocean_20_ku",
        "epoch_ns": "epoch_ocean_20_ku",
        "range": "range_ocean_20_ku",
        "tracker_range_m": "tracker_range_20_ku",
        "alt_m": "alt_20_ku",
        "n_iter": "number_of_iterations_ocean_20_ku",
        "misfit": "mqe_ocean_20_ku",
        # 'l2norm': 'Misfit_20Hz',
        # 'p_out_w': 'Pout_20Hz',
        # 'mss': 'MSS_20Hz',
    },
}

data_vars_cs = {
    "l1b": {
        "wf": "pwr_waveform_20_ku",
        "time": "time_20_ku",
        "lat_rad": "lat_20_ku",
        "lon_rad": "lon_20_ku",
        "alt_m": "alt_20_ku",
        "h_rate_m_per_s": "orb_alt_rate_20_ku",
        # 'Vs_m_per_s': 'sat_vel_vec_20_ku',
        "Vs_m_per_s": "sat_vel_vec_20_ku",
        "ksix_rad": "off_nadir_pitch_angle_str_20_ku",  # pitch
        "ksiy_rad": "off_nadir_roll_angle_str_20_ku",  # roll
        "beam_ang_stack_rad": "beam_dir_vec_20_ku",
        "epoch_ref_gate": "epoch_ref_gate_const",
        "tracker_range_m": "window_del_20_ku",
    },
    # #GOPR
    # 'l1b': {
    #     'wf': 'pwr_waveform_20_hr_ku',
    #     'time': 'time_20_hr_ku',
    #     'lat_rad': 'lat_20_hr_ku',
    #     'lon_rad': 'lon_20_hr_ku',
    #     'alt_m': 'alt_20_hr_ku',
    #     'h_rate_m_per_s': 'orb_alt_rate_20_hr_ku',
    #     # 'Vs_m_per_s': 'sat_vel_vec_20_hr_ku',
    #     'Vs_m_per_s': 'sat_vel_vec_20_hr_ku',
    #     'ksix_rad': 'off_nadir_pitch_angle_str_20_hr_ku',  # pitch
    #     'ksiy_rad': 'off_nadir_roll_angle_str_20_hr_ku',  # roll
    #     'beam_ang_stack_rad': 'beam_dir_vec_20_hr_ku',
    #     'epoch_ref_gate': 'epoch_ref_gate_const',
    #     'tracker_range_m': 'window_del_20_hr_ku',
    # },
    "l2": {
        "time": "time_20_ku",
        "lat_rad": "lat_20_ku",
        "lon_rad": "lon_20_ku",
        "swh": "swh_ocean_20_ku",
        "swh_qual": "swh_ocean_qual_20_ku",
        "Pu_W": "const",
        # 'epoch_ns': 'const',
        "range": "range_ocean_20_ku",
        # 'tracker_range_m': 'window_del_20_hr_ku',
        "alt_m": "alt_20_ku",
        "n_iter": "const",
        "misfit": "mqe_ocean_20_ku",
        # gpod
        # 'time': 'time_counter_20Hz',
        # 'lat_rad': 'latitude_20Hz',
        # 'lon_rad': 'longitude_20Hz',
        # 'swh': 'SWH_20Hz',
        # 'swh_qual': 'Misfit_20Hz',
        # 'Pu_W': 'Pout_20Hz',
        # 'epoch_ns': 'Epoch_20Hz',
        # 'range': 'Range_Unc_20Hz',
        # 'tracker_range_m': 'Window_Delay_20Hz',
        # 'alt_m': 'altitude_20Hz',
        # 'n_iter': 'Iterations_Count_20Hz',
        # 'misfit': 'Misfit_20Hz',
    },
}

data_vars_gpod = {
    "l1b": {
        "wf": "SAR_Echo_Data",
        "time": "time_counter_20Hz",
        "lat_rad": "latitude_20Hz",
        "lon_rad": "longitude_20Hz",
        "alt_m": "altitude_20Hz",
        "h_rate_m_per_s": "altitude_rate_20Hz",
        "Vs_m_per_s": "satellite_velocity_20Hz",
        "ksix_rad": "pitch_mispointing_20Hz",  # pitch
        "ksiy_rad": "roll_mispointing_20Hz",  # roll
        "beam_ang_stack_rad": "Stack_Boresight_Beam_Angle",
        "epoch_ref_gate": "Epoch_Reference_Gate",
        "tracker_range_m": "Window_Delay_20Hz",
        "rip": "Substack_RIP_Data",
        "n_looks_stack": "NLook_20Hz",
        "oversampling_factor": "const",
        "thermal_noise": "ThN_20Hz",
    },
    "l2": {
        "time": "time_counter_20Hz",
        "lat_rad": "latitude_20Hz",
        "lon_rad": "longitude_20Hz",
        "swh": "SWH_20Hz",
        "swh_qual": "Misfit_20Hz",
        "Pu_W": "Pout_20Hz",
        "epoch_ns": "Epoch_20Hz",
        "range": "Range_Unc_20Hz",
        "tracker_range_m": "Window_Delay_20Hz",
        "alt_m": "altitude_20Hz",
        "n_iter": "Iterations_Count_20Hz",
        "misfit": "Misfit_20Hz",
    },
}


data_vars_s6 = {
    "l1b": {
        "wf": "power_waveform",
        "time": "time",
        "lat_rad": "latitude",
        "lon_rad": "longitude",
        "alt_m": "altitude",
        "h_rate_m_per_s": "altitude_rate",
        "Vs_m_per_s": "velocity_vector",
        "ksix_rad": "off_nadir_pitch_angle_pf",  # pitch
        "ksiy_rad": "off_nadir_roll_angle_pf",  # roll
        "beam_ang_stack_rad": "look_angle_start",
        "stack_mask_start_stop": "stack_mask_start_stop",
        "epoch_ref_gate": "epoch_ref_gate_const",
        "tracker_range_m": "tracker_range_calibrated",
        "pri_hz": "pulse_repetition_interval",
    },
    "l2": {
        "time": "time",
        "lat_rad": "latitude",
        "lon_rad": "longitude",
        "swh": "swh_ocean",
        "swh_qual": "swh_ocean_qual",
        "Pu_W": "amplitude_ocean",
        "epoch_ns": "epoch_ocean",
        "range": "range_ocean",
        "tracker_range_m": "tracker_range_calibrated",
        "alt_m": "altitude",
        "n_iter": "num_iterations_ocean",
        "misfit": "mqe_ocean",
    },
}

data_vars_dart = {
    "l1b": {
        "wf": "power_waveform_pseudo_dda",
        "time": "time",
        "lat_rad": "latitude",
        "lon_rad": "longitude",
        "alt_m": "altitude",
        "h_rate_m_per_s": "altitude_rate",
        "Vs_m_per_s": "velocity_vector",
        "ksix_rad": "off_nadir_pitch_angle_pf",  # pitch
        "ksiy_rad": "off_nadir_roll_angle_pf",  # roll
        "beam_ang_stack_rad": "look_angles",
        "stack_mask_start_stop": "stack_mask_start_stop",
        "epoch_ref_gate": "epoch_ref_gate_var",
        "tracker_range_m": "tracker_range_calibrated",
        "pri_hz": "pulse_repetition_interval",
    },
    "l1b_ffsar": {
        "wf": "power_waveform",
        "time": "time",
        "lat_rad": "latitude",
        "lon_rad": "longitude",
        "alt_m": "altitude",
        "h_rate_m_per_s": "altitude_rate",
        "Vs_m_per_s": "velocity_vector",
        "ksix_rad": "off_nadir_pitch_angle_pf",  # pitch
        "ksiy_rad": "off_nadir_roll_angle_pf",  # roll
        "beam_ang_stack_rad": "look_angles",
        "stack_mask_start_stop": "stack_mask_start_stop",
        "epoch_ref_gate": "epoch_ref_gate_var",
        "tracker_range_m": "tracker_range_calibrated",
        "pri_hz": "pulse_repetition_interval",
    },
    "l2": {
        "time": "time",
        "lat_rad": "latitude",
        "lon_rad": "longitude",
        "swh": "swh_ocean",
        "swh_qual": "swh_ocean_qual",
        # 'Pu_W': 'amplitude_ocean',
        # 'epoch_ns': 'epoch_ocean',
        "range": "range_ocean",
        "tracker_range_m": "tracker_range_calibrated",
        "alt_m": "altitude",
        # 'n_iter': 'num_iterations_ocean',
        # 'misfit': 'mqe_ocean',
    },
}

data_vars_retracker = {
    "l2": {
        "time": "time",
        "lat_rad": "latitude",
        "lon_rad": "longitude",
        "swh": "swh",
        "swh_qual": "swh_qual",
        "Pu_W": "Pu",
        "epoch_ns": "epoch",
        "range": "range",
        "tracker_range_m": "tracker_range",
        "alt_m": "altitude",
        # 'n_iter': '',
        "misfit": "misfit",
    }
}


def get_last_ind(n_offset, n_inds, n_total_length):
    """Returns last (0-based) index (including it) by considering n_offset, n_inds and the maximal ind (end of file)

    :param n_offset: start index
    :param n_inds: number of indices
    :param n_total_length: maximum number of samples
    :return: index number
    """
    if n_inds == 0:
        ind_last = n_total_length - 1
    elif n_offset + n_inds >= n_total_length:
        ind_last = n_total_length - 1
    else:
        ind_last = n_offset + n_inds - 1

    if ind_last > n_total_length:
        raise RuntimeError(
            f"get_last_ind: last_ind ({ind_last}) exceeds total length ({n_total_length})"
        )

    return ind_last


def convert_eum_mqe_to_misfit(mqe):
    return np.sqrt(100 * np.sqrt(mqe))


def _read_dataset_vars_from_ds(
    *,
    nc_filename,
    data_var_names,
    n_offset=0,
    n_inds=0,
    do_interp_dist2coast=False,
    group=None,
):
    # hack required because some nc files have issues with decoding of times
    try:
        ds = xr.open_dataset(nc_filename, group=group)
    except Exception:
        ds = xr.open_dataset(nc_filename, group=group, decode_times=False)

    last_ind = get_last_ind(
        n_offset=n_offset,
        n_inds=n_inds,
        n_total_length=ds[data_var_names["time"]].shape[0],
    )

    def get_data_var(_v):
        if "const" in _v:
            var = None
        elif ds[_v].size == 1:
            var = float(ds[_v].values)
        else:
            var = ds[_v][n_offset : last_ind + 1].values

        # general conversions
        if (
            "lat" in _v.lower()
            or "lon" in _v.lower()
            or "off_nadir" in _v.lower()
            or "beam" in _v.lower()
        ):
            is_deg_unit = (
                hasattr(ds[_v], "units") and "deg" in str(ds[_v].units).lower()
            )
            # is_rad_unit = (hasattr(ds[_v], 'units') and 'rad' in ds[_v].units)
            if is_deg_unit or (
                not is_deg_unit and any((abs(ds[_v]) > np.pi / 2).values.ravel())
            ):
                var = np.radians(var)

        return var

    dataset = {}
    for k, v in data_var_names.items():
        if v in dir(ds) or "const" in v:
            is_eumetsat = (
                ("contact" in dir(ds) and "eumetsat" in ds.contact)
                or "EUM" in str(nc_filename)
            ) and "processor" not in dir(ds)
            is_gpod = (
                "NetCDF_CreatedBy" in dir(ds) and "SARvatore" in ds.NetCDF_CreatedBy
            )
            is_dart = "title" in dir(ds) and "dart" in ds.title.lower()
            is_s3 = "S3" in str(nc_filename)
            is_cs = "/CS" in str(nc_filename)
            is_cs_gopr = is_cs and "GOPR" in str(nc_filename)
            is_s6 = "S6" in str(nc_filename)

            # provider-specific conversions
            # GPOD
            if is_gpod and "oversampling_factor" in k:
                # zero_pad_factor = 2.0 if hasattr(ds, 'L1b_HISTORY') and 'Zero_Padding_Flag = Y' in ds.L1b_HISTORY else 1.0
                # data_var = zero_pad_factor * wf_oversampling_factor
                wf_oversampling_factor = (
                    float(ds.WAVEFORM_OVERSAMPLING_FACTOR)
                    if hasattr(ds, "WAVEFORM_OVERSAMPLING_FACTOR")
                    else 1
                )
                data_var = wf_oversampling_factor
            elif is_gpod and "Window_Delay_20Hz" in v:
                nat_mask = np.isnat(get_data_var(v))
                # convert window delay from ns to distance in m
                data_var = get_data_var(v).astype(float) * 1e-9 * CONST_C / 2
                data_var[nat_mask] = np.nan  #
            elif is_gpod and "Epoch_20Hz" in v:
                # wf_oversampling_factor = float(ds.WAVEFORM_OVERSAMPLING_FACTOR) if hasattr(ds, 'WAVEFORM_OVERSAMPLING_FACTOR') else 1
                # data_var = wf_oversampling_factor * get_data_var(v)  #
                # convert from s in ns
                data_var = get_data_var(v)  # convert from s in ns
            elif is_gpod and "time" in v and not is_cs:
                data_var = np.datetime64(ds.SENSING_START_UTC_TIME[:-1]) + get_data_var(
                    v
                )
            elif is_gpod and "time" in v and is_cs:
                dt_str = ds.SENSING_START[1:]
                dt_str = dt_str.replace(dt_str[3:6], dt_str[3:6].lower().title())
                data_var = np.datetime64(
                    datetime.strptime(dt_str, "%d-%b-%Y %H:%M:%S.%f")
                ) + get_data_var(v)
            elif is_gpod and "swh_qual" in k and "Misfit_20Hz" in v:
                # as defined in Dinardo2020, 3.4 (True=bad)
                data_var = get_data_var(v) > 4
            elif is_gpod and "Epoch_Reference_Gate" in v:
                # GPOD is 1-based, we need 0-based
                data_var = get_data_var(v) - 1
            # EUMETSAT or DeDop
            elif is_eumetsat and "lon" in k:
                # df.loc[df.lon >= 180.0, 'lon'] -= 360.0
                data_var = get_data_var(v)
                inds_to_convert = np.argwhere(get_data_var(v) > np.pi)
                data_var[inds_to_convert] -= 2 * np.pi
            elif is_eumetsat and "epoch_ref_gate_const" in v and is_s3:
                data_var = (
                    44.0 - 1
                )  # fixed, info from Bruno Lucas, Mail conversation from 6th of April
            elif is_eumetsat and "epoch_ref_gate_const" in v and is_s6:
                zp_factor = 2
                data_var = 128 * zp_factor
            elif (is_eumetsat) and "Vs" in k and is_s3:
                data_var = np.sqrt(
                    get_data_var("x_vel_l1b_echo_sar_ku") ** 2
                    + get_data_var("y_vel_l1b_echo_sar_ku") ** 2
                    + get_data_var("z_vel_l1b_echo_sar_ku") ** 2
                )
            elif (is_eumetsat or is_dart) and "Vs" in k and is_s6:
                v = get_data_var("velocity_vector")
                data_var = np.apply_along_axis(
                    lambda vel: np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2), 1, v
                )
            elif is_dart and "time" in v:
                data_var = np.datetime64(
                    ds.first_measurement_time[:-1]
                ) + pd.to_timedelta(get_data_var(v), "s")
            # CS
            elif is_cs and "epoch_ref_gate_const" in v:
                data_var = 128.0
            elif is_cs and "beam_ang_stack" in k:
                angle_start = (
                    get_data_var("dop_angle_start_20_hr_ku")
                    if is_cs_gopr
                    else get_data_var("dop_angle_start_20_ku")
                )
                angle_stop = (
                    get_data_var("dop_angle_stop_20_hr_ku")
                    if is_cs_gopr
                    else get_data_var("dop_angle_stop_20_ku")
                )
                n_beams = (
                    get_data_var("stack_number_after_weighting_20_hr_ku").astype(int)
                    if is_cs_gopr
                    else get_data_var("stack_number_after_weighting_20_ku").astype(int)
                )

                def func(start, stop, num):
                    arr = np.full(200, np.nan)
                    arr[:num] = np.linspace(start, stop, num) - np.pi / 2
                    return arr

                data_var = np.array(
                    [
                        func(start, stop, num)
                        for start, stop, num in zip(angle_start, angle_stop, n_beams)
                    ]
                )
            elif is_cs and "Vs" in k:
                data_var = np.sqrt(
                    get_data_var(v)[0, 0] ** 2
                    + get_data_var(v)[0, 1] ** 2
                    + get_data_var(v)[0, 2] ** 2
                )
            elif is_cs and "ksi" in k:
                data_var = get_data_var(v)
            elif is_cs and "tracker" in k:
                # data_var = get_data_var(v).astype(float) * 1e9
                data_var = get_data_var(v).astype(float) * 1e-9 * CONST_C / 2
            elif is_cs and "epoch" in k:
                trk_range = (
                    get_data_var("window_del_20_hr_ku")
                    if is_cs_gopr
                    else get_data_var("window_del_20_ku")
                )
                tracker_range_m = trk_range.astype(float) * 1e-9 * CONST_C / 2
                range_m = (
                    get_data_var("range_ocean_20_hr_ku")
                    if is_cs_gopr
                    else get_data_var("range_ocean_20_ku")
                )
                range_res_m = (1 / 320e6) * CONST_C / 2
                data_var = (
                    (range_m - tracker_range_m) * range_res_m * (1 / 320e6) / 1e-9
                )
            elif is_cs and "Pu_W" in k or "n_iter" in k:
                data_var = np.full(n_inds, np.nan)
            # EUMETSAT
            elif is_eumetsat and "wf" in k and is_s6:
                # data_var = get_data_var(v) * get_data_var('waveform_scale_factor')
                data_var = get_data_var(v)
            elif is_dart and "beam_ang_stack" in k:
                data_var = get_data_var(v) + np.pi / 2
            elif is_eumetsat and "beam_ang_stack" in v and is_s3:
                data_var = get_data_var(v) + np.pi / 2
            elif is_eumetsat and "beam_ang_stack" in k and is_s6:
                n_looks_max = int(ds.dims["looks"])

                angle_start = get_data_var("look_angle_start")
                angle_stop = get_data_var("look_angle_stop")
                num_looks = np.min(
                    [
                        get_data_var("num_looks_start_stop"),
                        get_data_var("num_looks_multilooking"),
                    ],
                    axis=0,
                ).astype(int)
                angle_per_look_rad = (angle_stop - angle_start) / (num_looks - 1)

                def func(start, num, angle_per_look, n_looks_max):
                    arr = np.full(n_looks_max, np.nan)
                    arr[:num] = start + np.arange(num) * angle_per_look + np.pi / 2
                    return arr

                angs = np.array(
                    [
                        func(start, num, angle_per_look, n_looks_max)
                        for start, num, angle_per_look in zip(
                            angle_start, num_looks, angle_per_look_rad
                        )
                    ]
                )

                data_var = np.fliplr(angs)
            elif is_eumetsat and is_s6 and "wf" in k:
                data_var = np.multiply(
                    get_data_var(v),
                    get_data_var("waveform_scale_factor")[:, np.newaxis],
                )
            elif is_eumetsat and "mqe_ocean_20_ku" in v:
                data_var = convert_eum_mqe_to_misfit(get_data_var(v))
            elif is_eumetsat and "epoch_ocean_20_ku" in v:
                data_var = get_data_var(v) / 1e-9  # convert from s in ns
            elif is_eumetsat and "alt" in k and is_s3:
                # for L1BS files, subtract center-of-gravity to antenna
                # distance from altitude
                if "1BS" in ds.title:
                    cog = get_data_var("cog_cor_l1b_echo_sar_ku")
                    data_var = get_data_var(v) - cog
                else:
                    data_var = get_data_var(v)
            elif is_s6 and "ksi" in k:
                data_var = np.radians(get_data_var(v))
            # all
            elif is_eumetsat and "misfit" in k:
                mqe_var = [v for k, v in data_var_names.items() if "mqe" in v][0]
                data_var = convert_eum_mqe_to_misfit(get_data_var(mqe_var)) > 4
            else:
                data_var = get_data_var(v)

            dataset[k] = data_var

    if len(dataset) != len(data_var_names):
        missing = [d for d in list(data_var_names) if d not in list(dataset)]
        raise RuntimeError(
            "Not all fields were found in datasets. Missing: {}".format(
                ",".join(missing)
            )
        )

    # add RIP if available (first found occurrence taken)
    rip_var_list = [d for d in ds.variables if "rip" in d.lower()]
    if rip_var_list:
        dataset["rip"] = get_data_var(rip_var_list[0])

    # add entropy, pulse_peakiness etc.
    if "wf" in dataset:
        wf_norm = (dataset["wf"].T / np.max(dataset["wf"], axis=1).T).T
        dataset["entropy"] = np.apply_along_axis(
            lambda wf: -np.nansum(np.square(wf) * np.log2(np.square(wf))), 1, wf_norm
        )
        dataset["pulse_peakiness"] = np.nanmax(dataset["wf"], axis=1) / np.nansum(
            dataset["wf"], axis=1
        )

    # dist2coast
    dataset["dist2coast"] = get_dist_pacioos(
        latarr=np.degrees(dataset["lat_rad"]),
        lonarr=np.degrees(dataset["lon_rad"]),
        do_interp=do_interp_dist2coast,
    )
    # dirty hack: we assume land-water-transitions are set nan -> set them to
    # 0.0 so that we retrack them
    dataset["dist2coast"][np.isnan(dataset["dist2coast"])] = 0.0

    # add ascending or descending track boolean var
    if is_gpod and "PASS_DIRECTION_START" in ds.attrs:
        dataset["ascending"] = "ascending" == ds.PASS_DIRECTION_START
    elif dataset["lat_rad"].shape[0] > 1:
        dataset["ascending"] = bool(dataset["lat_rad"][1] > dataset["lat_rad"][0])

    if "record_ind" in ds:
        dataset["record_inds"] = ds.record_ind.values[n_offset : last_ind + 1]
    else:
        dataset["record_inds"] = np.arange(n_offset, last_ind + 1)

    return dataset


def get_model_param_obj_from_l1b_data(dataset: dict, ind: int):
    d = {
        k: (v if isinstance(v, (bool, float, int)) or len(v) <= 1 else v[ind])
        for k, v in dataset.items()
        if v is not None
    }
    return ModelParameter(**d)


def get_subset_dataset(dataset: dict, ind_offset: int, n_inds: int = 1):
    d = {}

    for k, v in dataset.items():
        if np.isscalar(v):
            d[k] = v
        elif isinstance(v, (np.ndarray, np.generic)):
            if v.size == 1:
                d[k] = v[0]
            elif v.size > 1 and len(v.shape) > 1:
                d[k] = v[ind_offset : ind_offset + n_inds, :]
            elif v.size > 1 and len(v.shape) == 1:
                d[k] = v[ind_offset : ind_offset + n_inds]

                d[k] = d[k][0] if d[k].size == 1 else d[k]
            else:
                RuntimeError("Invalid type")

            if len(d[k].shape) > 1 and d[k].shape[0] == 1:
                d[k] = d[k].squeeze()
        else:
            RuntimeError("Invalid type")

    return d


def gen_subset_dataset(dataset, rel_inds):
    for ind in rel_inds:
        yield get_subset_dataset(dataset, ind)


def gen_model_param_obj_from_l1b_data(dataset, rel_inds):
    for ind in rel_inds:
        yield get_model_param_obj_from_l1b_data(dataset, ind)


def get_nc_src_dest_file_list(l1b_source, nc_dest_dir):
    nc_dest_dir = nc_dest_dir if nc_dest_dir else Path(tempfile.gettempdir())
    abs_dest_dir = nc_dest_dir.resolve()

    if not isinstance(l1b_source, list) and os.path.isdir(l1b_source):
        src_files = list(l1b_source.rglob("/*.nc"))
        src_files_from_base = [f.parent for f in src_files]
        dest_files = [abs_dest_dir / f for f in src_files_from_base]
    elif not isinstance(l1b_source, list):
        src_files = list(l1b_source)

        # add further level directory if files without any unique identifier,
        # e.g. measurement_l1bs.nc
        src_files_basename = (
            src_files[0].parent.stem
            if "measurement" in src_files[0].stem and "SEN6" not in src_files[0].stem
            else src_files[0].stem
        )

        if "SEN" in src_files_basename and "measurement" in src_files_basename:
            dest_files = [
                abs_dest_dir / f'{f.name.split(".SEN6.")[0]}.nc' for f in src_files
            ]
        elif "S3" not in src_files_basename and "measurement" in src_files_basename:
            dest_files = []
            for f in src_files:
                track_id = f.parent.parent.name
                file_id = f.parent.name
                dest_files.append(
                    abs_dest_dir / track_id / file_id / src_files_basename
                )
        elif "SEN" in str(src_files[0].parent.name):
            dest_files = [abs_dest_dir / f.parent.name / f.name for f in src_files]
        elif "SEN" in src_files[0].parent.name:
            dest_files = [abs_dest_dir / f.parent.name / f.name for f in src_files]
        else:
            dest_files = [abs_dest_dir / f.name for f in src_files]
    else:
        src_files = l1b_source
        if "measurement" in str(src_files[0]):
            dest_files = [
                abs_dest_dir / (f.parent.name.replace(".SEN6", "") + ".nc")
                for f in src_files
            ]
        else:
            dest_files = [abs_dest_dir / f.name for f in src_files]

    return src_files, dest_files


def get_nc_dir_df(nc_files):
    all_nc_files = sorted(nc_files.rglob("*.nc"))
    is_l2 = "_l2_" in str(all_nc_files[0]).lower()

    def convert_dt(dt_str):
        return np.datetime64(datetime.strptime(dt_str, "%Y%m%dT%H%M%S"))

    if is_l2:
        cycles = [int(f.name.split("_")[8]) for f in all_nc_files]
        passes = [int(f.name.split("_")[9]) for f in all_nc_files]
        start_dates = [convert_dt(f.name.split("_")[10]) for f in all_nc_files]
        end_dates = [convert_dt(f.name.split("_")[11]) for f in all_nc_files]
    else:
        cycles = [int(f.name.split("_")[13]) for f in all_nc_files]
        passes = [int(f.name.split("_")[14]) for f in all_nc_files]
        start_dates = [convert_dt(f.name.split("_")[9]) for f in all_nc_files]
        end_dates = [convert_dt(f.name.split("_")[10]) for f in all_nc_files]

    df = pd.DataFrame(
        {
            "file": all_nc_files,
            "cycle": cycles,
            "ppass": passes,
            "start_date": start_dates,
            "end_date": end_dates,
        }
    )

    return df
