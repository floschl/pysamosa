import numpy as np
import pytest

from pysamosa.common_types import SENSOR_SETS_DEFAULT_S3
from pysamosa.retracker_helpers import (
    gen_sliding_window_from_seq,
    get_dynamic_first_guess_epochs,
    get_inds_sliding_window,
    get_pointwise_product,
    get_tracker_shift_n_samps,
    shift_waveform_by_n,
)


@pytest.mark.parametrize(
    "n_offset, n_inds",
    [
        (25800, 100),
    ],
)
def test_get_tracker_shift_n_samps(n_offset, n_inds, s3_eum_l1b):
    l1b_data = s3_eum_l1b(n_offset=n_offset, n_inds=n_inds, file_id="s3_0")

    n_shifts = get_tracker_shift_n_samps(
        tracker_range_m=l1b_data["tracker_range_m"][:20],
        alt_m=l1b_data["alt_m"][:20],
        t0_sampling_gate_ns=SENSOR_SETS_DEFAULT_S3.B_r_Hz,
    )

    assert len(n_shifts) == 20
    assert all(n_shifts < 3)


@pytest.mark.parametrize(
    "n_offset, n_inds",
    [
        (25800, 100),
    ],
)
def test_get_dynamic_first_guess_epochs_sam_coast(n_offset, n_inds, s3_eum_l1b):
    l1b_data = s3_eum_l1b(n_offset=n_offset, n_inds=n_inds, file_id="s3_0")

    fg_epoch = get_dynamic_first_guess_epochs(
        wfs=l1b_data["wf"],
        tracker_range=l1b_data["tracker_range_m"],
        alt_m=l1b_data["alt_m"],
        bu_bw_Hz=SENSOR_SETS_DEFAULT_S3.B_r_Hz,
        fg_epoch_adjacent_meas=40,
        # n_procs=None,
        n_procs=1,
    )

    assert len(fg_epoch) == l1b_data["wf"].shape[0]


@pytest.mark.parametrize(
    "n_offset, n_inds",
    [
        (25800, 1000),
    ],
)
def test_get_dynamic_first_guess_epochs_single_multcore(n_offset, n_inds, s3_eum_l1b):
    l1b_data = s3_eum_l1b(n_offset=n_offset, n_inds=n_inds, file_id="s3_0")

    fg_epoch = get_dynamic_first_guess_epochs(
        wfs=l1b_data["wf"],
        tracker_range=l1b_data["tracker_range_m"],
        alt_m=l1b_data["alt_m"],
        bu_bw_Hz=SENSOR_SETS_DEFAULT_S3.B_r_Hz,
        fg_epoch_adjacent_meas=40,
        n_procs=1,
    )

    fg_epoch_multcore = get_dynamic_first_guess_epochs(
        wfs=l1b_data["wf"],
        tracker_range=l1b_data["tracker_range_m"],
        alt_m=l1b_data["alt_m"],
        bu_bw_Hz=SENSOR_SETS_DEFAULT_S3.B_r_Hz,
        fg_epoch_adjacent_meas=40,
        n_procs=12,
    )

    assert len(fg_epoch) == l1b_data["wf"].shape[0]
    assert np.array_equal(fg_epoch, fg_epoch_multcore)


@pytest.mark.parametrize(
    "n_offset, n_inds",
    [
        (25800, 125),
    ],
)
def test_get_dynamic_first_guess_epochs_samplus(n_offset, n_inds, s3_eum_l1b):
    l1b_data = s3_eum_l1b(n_offset=n_offset, n_inds=n_inds, file_id="s3_0")

    fg_epoch = get_dynamic_first_guess_epochs(
        wfs=l1b_data["wf"],
        tracker_range=l1b_data["tracker_range_m"],
        alt_m=l1b_data["alt_m"],
        bu_bw_Hz=SENSOR_SETS_DEFAULT_S3.B_r_Hz,
    )

    assert len(fg_epoch) == l1b_data["wf"].shape[0]
    assert np.isclose(np.mean(fg_epoch), 44, atol=2)


def test_get_inds_sliding_window():
    exp_res = [
        ([0, 1, 2], 0),
        (
            [0, 1, 2],
            1,
        ),
        ([1, 2, 3], 1),
        ([2, 3, 4], 1),
        ([3, 4, 5], 1),
        ([3, 4, 5], 2),
    ]
    res = get_inds_sliding_window(n_total_len=6, n=3)
    assert res == exp_res

    exp_res = [
        ([0, 1, 2], 0),
        ([0, 1, 2], 1),
        ([1, 2, 3], 1),
        ([2, 3, 4], 1),
        ([2, 3, 4], 2),
    ]
    res = get_inds_sliding_window(n_total_len=5, n=3)
    assert res == exp_res

    exp_res = [
        ([0, 1, 2, 3], 0),
        ([0, 1, 2, 3], 1),
        ([0, 1, 2, 3], 2),
        ([1, 2, 3, 4], 2),
        ([1, 2, 3, 4], 3),
    ]
    res = get_inds_sliding_window(n_total_len=5, n=4)
    assert res == exp_res

    exp_res = [
        ([0, 1, 2], 0),
        ([0, 1, 2], 1),
        ([0, 1, 2], 2),
    ]
    res = get_inds_sliding_window(n_total_len=3, n=4)
    assert res == exp_res

    exp_res = [
        ([0, 1, 2, 3], 0),
        ([0, 1, 2, 3], 1),
        ([0, 1, 2, 3], 2),
        ([1, 2, 3, 4], 2),
        ([2, 3, 4, 5], 2),
        ([3, 4, 5, 6], 2),
        ([4, 5, 6, 7], 2),
        ([5, 6, 7, 8], 2),
        ([6, 7, 8, 9], 2),
        ([6, 7, 8, 9], 3),
    ]
    res = get_inds_sliding_window(n_total_len=10, n=4)
    assert res == exp_res

    exp_res = [
        ([0, 1, 2, 3, 4, 5], 0),
        ([0, 1, 2, 3, 4, 5], 1),
        ([0, 1, 2, 3, 4, 5], 2),
        ([0, 1, 2, 3, 4, 5], 3),
        ([1, 2, 3, 4, 5, 6], 3),
        ([2, 3, 4, 5, 6, 7], 3),
        ([3, 4, 5, 6, 7, 8], 3),
        ([4, 5, 6, 7, 8, 9], 3),
        ([4, 5, 6, 7, 8, 9], 4),
        ([4, 5, 6, 7, 8, 9], 5),
    ]
    res = get_inds_sliding_window(n_total_len=10, n=6)
    assert res == exp_res


def test_genslidingwindow():
    x = np.array([1] * 2 + [2] * 4 + [3] * 2 + [7])
    n = 4

    exp = [
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 2, 2, 2, 2],
        [2, 2, 2, 2, 3],
        [2, 2, 2, 3, 3],
        [2, 2, 3, 3, 7],
        [2, 2, 3, 3, 7],
        [2, 2, 3, 3, 7],
    ]

    exp_center = np.array([0, 1, 2, 2, 2, 2, 2, 3, 4])

    for i, (wnd, center_sample) in enumerate(gen_sliding_window_from_seq(x, n)):
        assert list(wnd) == exp[i]
        assert center_sample == exp_center[i]


def test_shift_waveform_by_n():
    X = np.tile(np.arange(10), 4).reshape(4, 10)

    res = shift_waveform_by_n(X, (0, 2, 3, -2))
    assert np.array_equal(
        res,
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
                [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 0, 0],
            ]
        ),
    )


def test_get_pointwise_product():
    X = np.tile(np.arange(10), 4).reshape(4, 10)

    assert all(
        get_pointwise_product(X)
        == np.array([0, 4, 16, 36, 64, 100, 144, 196, 256, 324])
    )
    assert all(get_pointwise_product(X, 1) == np.array([285, 285, 285, 285]))
