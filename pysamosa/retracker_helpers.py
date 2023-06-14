from collections import deque
from concurrent import futures
from itertools import islice, repeat

import numpy as np

from pysamosa.conf_params import CONST_C
from pysamosa.data_access import (
    _read_dataset_vars_from_ds,
    get_model_param_obj_from_l1b_data,
    get_subset_dataset,
)
from pysamosa.retracker import SamosaRetracker


def get_tracker_shift_n_samps(
    tracker_range_m: np.ndarray, alt_m: np.ndarray, t0_sampling_gate_ns: float
) -> np.ndarray:
    """
    Calulates the number of samples to shift for range-aligning multiple waveforms.

    :param tracker_range_m: distance between altimeter and the receiving window
    :param alt_m: altitude of altimeter
    :param t0_sampling_gate_ns: sampling time of one gate sample in the receiving window
    :return: an np.ndarray containing the integer number of samples to shift
    """
    t_diff = np.subtract(alt_m, tracker_range_m) / CONST_C
    t_corr_no_bias = np.subtract(t_diff, np.nanmean(t_diff))

    n_shift_samps = np.round(np.divide(t_corr_no_bias, t0_sampling_gate_ns))
    n_shift_samps = np.nan_to_num(n_shift_samps)  # set nan values to a 0-shift

    return n_shift_samps.astype(int)


def custom_roll(arr, r_tup):
    m = np.asarray(r_tup)
    # need `copy`
    arr_roll = arr[:, [*range(arr.shape[1]), *range(arr.shape[1] - 1)]].copy()
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    result = np.lib.stride_tricks.as_strided(
        arr_roll, (*arr.shape, n), (strd_0, strd_1, strd_1)
    )

    return result[np.arange(arr.shape[0]), (n - m) % n]


def shift_waveform_by_n(wfs: np.ndarray, shifts: np.ndarray):
    dims_orig = wfs.shape
    if wfs.shape[0] != len(shifts):
        raise RuntimeError(
            "shifts vector must be of the same length as the number of waveforms. "
        )

    wfs = wfs.copy()

    wfs = np.pad(wfs, ((0, 0), (0, np.max(shifts))), mode="constant")
    res = custom_roll(wfs, shifts)[:, : dims_orig[1]]
    return res


def get_pointwise_product(wfs: np.ndarray, axis=0) -> np.ndarray:
    op = "ij,ij->j" if axis == 0 else "ij,ij->i"
    return np.einsum(op, wfs, wfs)


def get_inds_sliding_window(n_total_len, n):
    """Generates a sliding window with the absolute indices.

    :param n_total_len: length of total sequence lengths
    :param n: sliding window size
    :return:
    """

    def sliding_window(seq, n=2):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = list(islice(it, n))
        if len(result) == n:
            yield (result, n // 2)
        for elem in it:
            result = result[1:] + [
                elem,
            ]
            yield (result, n // 2)

    # pre- and append for n-1 times the same window to account for sliding
    # in/out effect
    if n_total_len > n:
        n_prepend = int(n // 2)
        n_append = n - (n_prepend + 1)
        windows_no_slidein = list(sliding_window(np.arange(n_total_len), n))
        windows = [
            *[(w, i) for i, w in enumerate([windows_no_slidein[0][0]] * n_prepend)],
            *windows_no_slidein,
            *[
                (w, (n // 2 + 1 + i))
                for i, w in enumerate([windows_no_slidein[-1][0]] * n_append)
            ],
        ]
    else:
        windows = [
            (win, i) for i, win in enumerate([list(range(n_total_len))] * n_total_len)
        ]

    return windows


def gen_sliding_window_from_seq(seq, n):
    """
    Generates a sliding window over an iterable sequence.

    :param seq: the iterable sequence the sliding window is generated from
    :param n: number of surrounding samples, the total sliding window size is n+1, n must be even-numbered
    :return: the sliding window of the current iteration
    """
    if n % 2:
        raise ValueError("n must be even-numbered. ")
    if len(seq) < n:
        raise ValueError("len(seq) must be larger than n. ")

    center_sample = 0

    half_window = n // 2
    n_total = n + 1
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n + 1)), maxlen=n_total)

    # sliding window in
    for i in range(half_window + 1):
        yield win, i
    center_sample = half_window

    # sliding inside
    for e in it:
        win.append(e)
        if center_sample < n // 2:
            center_sample += 1
        yield win, center_sample

    # sliding window out
    for i in range(center_sample + 1, n_total):
        yield win, i


def get_single_dynamic_first_guess_epochs(
    wfs: np.ndarray,
    tracker_range: np.ndarray,
    alt_m: np.ndarray,
    bu_bw_Hz,
    shifted_window_abs_inds_center: list,
):
    n_gates = wfs.shape[1]

    # shifted_window = shifted_window_abs_inds_center[0]
    rel_ind_window_center = shifted_window_abs_inds_center[1]

    # get shift in n samples that aligns all waveforms in terms of its
    # absolute ranges
    n_shifts = get_tracker_shift_n_samps(
        tracker_range_m=tracker_range, alt_m=alt_m, t0_sampling_gate_ns=1 / bu_bw_Hz
    )

    # if there is a very abrupt jump in the tracker range, then limit the
    # adjacent samples to have a maximum shift of half of the wf length
    mask_possible_shifts = np.abs((n_shifts - n_shifts[rel_ind_window_center])) < (
        n_gates // 3
    )

    # adjust possible indices
    rel_ind_window_center_possible = rel_ind_window_center - np.sum(
        ~mask_possible_shifts[0:rel_ind_window_center]
    )  # accounts for invalid tracker range jumps
    n_shifts = n_shifts[mask_possible_shifts]
    n_shifts -= int(np.mean(n_shifts))  # demean

    wfs_range_aligned = shift_waveform_by_n(wfs[mask_possible_shifts, :], -n_shifts)

    # wf_current_shifted = wfs_range_aligned[rel_ind_window_center_possible]

    # normalised range-aligned wf
    wfs_range_aligned_norm = np.apply_along_axis(
        lambda wf: wf / np.max(wf), 1, wfs_range_aligned
    )

    if all(
        np.apply_along_axis(
            lambda wf: np.allclose(wf[~np.isnan(wf)], 0.0),
            axis=1,
            arr=wfs_range_aligned_norm,
        )
    ):
        return 0, np.full((n_gates,), np.nan), np.full((n_gates,), np.nan)

    # if there are wfs with all-zero or all-nan, set them to 1 -> no effect on
    # pointwise product
    wfs_range_aligned_norm = np.apply_along_axis(
        lambda wf: np.ones(len(wf))
        if all(np.isnan(wf)) or np.isclose(np.mean(wf), 0.0, atol=1e-25)
        else wf,
        1,
        wfs_range_aligned_norm,
    )
    wfs_pw_prod = get_pointwise_product(wfs_range_aligned_norm)

    ind_max_prod = np.argmax(wfs_pw_prod)
    ind_max_adjacent_wfs = ind_max_prod

    new_fg_epoch = ind_max_adjacent_wfs + n_shifts[rel_ind_window_center_possible]

    return new_fg_epoch, wfs_range_aligned_norm, wfs_pw_prod


def get_dynamic_first_guess_epochs(
    wfs: np.ndarray,
    tracker_range: np.ndarray,
    alt_m: np.ndarray,
    bu_bw_Hz,
    fg_epoch_adjacent_meas=20,
    n_procs=1,
) -> np.ndarray:
    """Calculates the dynamic first-guess epochs based on the pointwise product of the n_adjacent_meas waveforms

    e.g. for n_adjacent_meas=20, for calculating the new fg epoch for waveform W_n:
    - account for alt_m_n - tracker_range_n -> shift W_n correspondingly
    - take  point-wise product of these tracker_range_n-aligned waveforms: W_n_-10, W_n_-9, ..., W_n, ..., W_n_9

    :param wfs: np.ndarray containing the waveforms (n_adjacent_samps, n_window_length)
    :param tracker_range: range of tracking window (L1B parameter)
    :param alt_m: altitude of satellite (L1B parameter)
    :param bu_bw_Hz: receiver bandwidth (L1B parameter)
    :param fg_epoch_adjacent_meas: number of adjacent measurements to take into account
    :param n_procs: number of processes, if None all available logical cores will be used
    :return: new first-guess epochs in gates, starting from 0
    """
    new_fg_epochs = np.ones(wfs.shape[0], dtype=int)
    n_gates = wfs.shape[1]

    if n_procs == 1:
        for abs_ind, shifted_window_abs_inds_center in enumerate(
            get_inds_sliding_window(wfs.shape[0], n=fg_epoch_adjacent_meas)
        ):
            wfs_window = wfs[shifted_window_abs_inds_center[0],]
            tracker_range_window = tracker_range[shifted_window_abs_inds_center[0]]
            alt_m_window = alt_m[shifted_window_abs_inds_center[0]]

            new_fg_epoch, _, _ = get_single_dynamic_first_guess_epochs(
                wfs=wfs_window,
                tracker_range=tracker_range_window,
                alt_m=alt_m_window,
                bu_bw_Hz=bu_bw_Hz,
                shifted_window_abs_inds_center=shifted_window_abs_inds_center,
            )

            if 0 < new_fg_epoch < n_gates:
                new_fg_epochs[abs_ind] = new_fg_epoch
            else:
                new_fg_epochs[abs_ind] = np.argmax(wfs[abs_ind])
    else:
        with futures.ProcessPoolExecutor(max_workers=n_procs) as pool:
            shifted_window_abs_inds_center = get_inds_sliding_window(
                wfs.shape[0], n=fg_epoch_adjacent_meas
            )

            def gen_subset_view(arr, shifted_window_abs_inds_center):
                for i in range(len(shifted_window_abs_inds_center)):
                    yield arr[shifted_window_abs_inds_center[i][0],]

            wfs_gen = gen_subset_view(wfs, shifted_window_abs_inds_center)
            alt_m_gen = gen_subset_view(alt_m, shifted_window_abs_inds_center)
            tracker_range_gen = gen_subset_view(
                tracker_range, shifted_window_abs_inds_center
            )

            for abs_ind, (new_fg_epoch, _, _) in enumerate(
                pool.map(
                    get_single_dynamic_first_guess_epochs,
                    wfs_gen,
                    tracker_range_gen,
                    alt_m_gen,
                    repeat(bu_bw_Hz),
                    shifted_window_abs_inds_center,
                    chunksize=100,
                )
            ):
                if 0 < new_fg_epoch < n_gates:
                    new_fg_epochs[abs_ind] = new_fg_epoch
                else:
                    new_fg_epochs[abs_ind] = np.argmax(wfs[abs_ind])

    return new_fg_epochs


def retrack_single_meas(*, sr, rp_sets, n_offset, nc_l1bs, datavars, sensor_sets):
    # calculate dynamic firt-guess epoch for a single measurement
    if rp_sets.do_dynamic_fg_epoch:
        # read n samples before the selected ones, required for estimation
        # dynamic first guess epoch
        n_inds_total = rp_sets.dynamic_fg_epoch_n_adjacent_meas
        n_pre = n_inds_total // 2

        l1b_data = _read_dataset_vars_from_ds(
            nc_filename=nc_l1bs,
            data_var_names=datavars,
            n_offset=n_offset - n_pre,
            n_inds=n_inds_total,
        )
        l1b_data["dynamic_fg_epoch"] = get_dynamic_first_guess_epochs(
            wfs=l1b_data["wf"],
            tracker_range=l1b_data["tracker_range_m"],
            alt_m=l1b_data["alt_m"],
            bu_bw_Hz=sensor_sets.B_r_Hz,
            fg_epoch_adjacent_meas=rp_sets.dynamic_fg_epoch_n_adjacent_meas,
        )
        l1b_data_single = get_subset_dataset(l1b_data, ind_offset=n_pre)
        model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=n_pre)
    else:
        l1b_data = _read_dataset_vars_from_ds(
            nc_filename=nc_l1bs, data_var_names=datavars, n_offset=n_offset, n_inds=1
        )
        l1b_data_single = get_subset_dataset(l1b_data, ind_offset=0)
        model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=0)

    # start fitting
    res_fit = sr.fit_wf(l1b_data_single=l1b_data_single, model_params=model_params)

    fg_epoch = (
        l1b_data_single["dynamic_fg_epoch"]
        if "dynamic_fg_epoch" in l1b_data_single
        else np.argmax(l1b_data_single["wf"])
    )
    fg_epoch = fg_epoch * sr.wf_sets.internal_oversampling_factor

    if sr.retrack_sets and sr.retrack_sets.normalise_wf_by_fg_region:
        wf_meas = res_fit["wf"] / SamosaRetracker.get_wf_max(
            res_fit["wf"], retrack_sets=sr.retrack_sets, fg_epoch=fg_epoch
        )
    else:
        wf_meas = res_fit["wf"] / np.max(res_fit["wf"])

    return wf_meas, res_fit, l1b_data_single, model_params, fg_epoch
