import numpy as np

from pysamosa.dist2coast import get_dist_pacioos

# def test_dist2coast_interp_on():
#     """Test for cross-checking the results and performance of different dist2coast sources. """
#
#     n_repl = 1
#     coords = {
#             'lat': np.array([54.6177, 42.40, 54.0886, 54.7985, 36.0579, 48.1411] * n_repl),
#             'lon': np.array([12.6001, -10.01, 8.8043, 7.1976, -5.3182, 11.5777] * n_repl),
#             'exp_dist': np.array([14.77, 78.00, 4.00, 69.24, 7.21, -288.11] * n_repl),
#     }
#
#     dist_pacioos = get_dist_pacioos(coords['lat'], coords['lon'], do_interp=True)
#     assert np.allclose(np.round(dist_pacioos, 2), coords['exp_dist'])
#
#     # dist_distcoast00 = get_dist_distcoast00(coords['lat'], coords['lon'])
#     # assert np.array_equal(np.round(dist_distcoast00, 2), coords['exp_dist'])
#
#     # assert np.allclose(dist_pacioos, dist_distcoast00, atol=1e-2)


def test_dist2coast_interp_off():
    """Test for cross-checking the results and performance of different dist2coast sources."""

    coords = {
        "lat": np.array(
            [
                -17.115001,
                -17.115,
                34.415000,
                34.415,
                33.825000,
                33.825,
                -61.185001,
                -61.185,
                -57.335000,
                -57.335,
                -8.869499,
                -8.869500,
                -53.546144,
                -53.546144,
                -57.335,
                -57.33499999,
                -16.665949,
                -16.66595,
                -57.654950,
                -57.65495,
                19.7175,
                19.7175,
                -13.046305,
                -13.046305,
            ]
        ),
        "lon": np.array(
            [
                107.738273,
                107.738273,
                128.728214,
                128.728214,
                128.385038,
                128.385037,
                66.382145,
                66.382145,
                75.772768,
                75.772768,
                110.906664,
                110.906663,
                82.169500,
                82.169500,
                75.772768,
                75.772768,
                107.921180,
                107.921180,
                75.119155,
                75.119155,
                121.519374,
                121.519373,
                109.340001,
                109.340000,
            ]
        ),
        "exp_dist": np.array(
            [
                759.0,
                759.0,
                29.0,
                29.0,
                78.0,
                78.0,
                692.0,
                692.0,
                482.0,
                482.0,
                69.0,
                69.0,
                557.0,
                557.0,
                482.0,
                482.0,
                718.0,
                718.0,
                506.0,
                506.0,
                29.0,
                29.0,
                486.0,
                486.0,
            ]
        ),
    }

    dist_pacioos = get_dist_pacioos(coords["lat"], coords["lon"], do_interp=False)
    print(dist_pacioos)

    assert np.array_equal(dist_pacioos, (np.round(coords["exp_dist"])).astype(int))
