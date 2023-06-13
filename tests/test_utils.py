import numpy as np
from pysamosa.utils import consecutive_regions_from_ind_list


def test_consecutive_regions_from_ind_list():
    a = [
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
    ]

    regions = consecutive_regions_from_ind_list(a)
    assert len(regions) == 1
    assert np.array_equal(a, regions[0])

    a = [2, 3, 4, 6, 7]
    regions = consecutive_regions_from_ind_list(a)
    assert len(regions) == 2
    assert np.array_equal(regions[0], [2, 3, 4])
    assert np.array_equal(regions[1], [6, 7])

    a = []
    regions = consecutive_regions_from_ind_list(a)
    assert len(regions) == 0


from pysamosa import dist2coast


def test_download_nc_file():
    dist2coast.download_dist2coast_nc()
