from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from pysamosa.utils import download_untar_file

DEFAULT_DUMMY_LAT = 48.1411
DEFAULT_DUMMY_LON = 11.5777


URL_D2C_FILE_SIZE = (
    "https://www.dropbox.com/s/1jhren60kmysr0w/dist2coast_1deg_merged.tar?dl=1",
    330857131,
)
D2C_DEST_PATH = Path(__file__).parent / ".data"
D2C_FILE_PATH = D2C_DEST_PATH / (Path(URL_D2C_FILE_SIZE[0]).stem + ".nc")


def download_dist2coast_nc():
    return download_untar_file(
        url=URL_D2C_FILE_SIZE[0],
        dest_path=D2C_DEST_PATH,
        expected_file_size=URL_D2C_FILE_SIZE[1],
    )


def get_dist_pacioos(latarr, lonarr, do_interp=True, pre_loaded_chunk_size=None):
    """
    Calculates distance to coast based on PacIOOS source.
    By now for ocean only.

    Online resource:
    - Ocean: http://www.pacioos.hawaii.edu/metadata/dist2coast_1deg_ocean.html
    - Land: http://www.pacioos.hawaii.edu/metadata/dist2coast_1deg_land.html

    :param latarr: latitude in decimal degrees, ndarray
    :param lonarr: longitude in decimal degrees, assumed to be within range (-180, 180] degrees, ndarray
    :param do_interp: if True, interpolation is active and distance to coast is calculated for coords outside 0.01-degree grid
    :return: distance to coast in km
    """
    # download dist2coast grid if not downloaded yet
    download_dist2coast_nc()

    try:
        _latarr = np.asarray(latarr)
        _lonarr = np.asarray(lonarr)
    except BaseException:
        raise ValueError("latarr, lonarr are not convertible to ndarrays. ")

    if _latarr.size != _lonarr.size:
        raise ValueError

    np.seterr(invalid="ignore")
    if ((-180 < _lonarr) & (_lonarr >= 180.0)).any():
        _lonarr[_lonarr >= 180.0] -= 360.0

    _step = 0.01
    _first_lat = 90.00
    _first_lon = -180.00

    with xr.open_dataset(
        D2C_FILE_PATH, engine="h5netcdf", chunks=pre_loaded_chunk_size
    ) as ds:
        if do_interp:

            def func_lat_ind_float(_lat):
                return -(_lat - _first_lat) / _step

            def func_lon_ind_float(_lon):
                return (_lon - _first_lon) / _step

            res = ndimage.map_coordinates(
                ds.dist,
                [func_lat_ind_float(latarr), func_lon_ind_float(lonarr)],
                order=1,
            )
        else:
            # trunc = lambda arr, n_dec: np.trunc(arr * 10 ** n_dec) / 10 ** n_dec
            avoid_round_error = 0.001
            lat_ind = (
                np.round(-(_latarr - _first_lat) / _step + avoid_round_error)
            ).astype(int)
            lon_ind = (
                np.round((_lonarr - _first_lon) / _step + avoid_round_error)
            ).astype(int)

            res = ds.dist.isel(
                lat=xr.DataArray(lat_ind % ds.dist.shape[0]),
                lon=xr.DataArray(lon_ind % ds.dist.shape[1]),
            ).values

    # insert nans where there are in the original latarr, lonarr
    lat_nan_inds = np.isnan(_latarr)
    lon_nan_inds = np.isnan(_lonarr)
    if np.isnan([latarr, lonarr]).any():
        res[lat_nan_inds | lon_nan_inds] = np.nan

    return res
