import os
from pathlib import Path

from tqdm import tqdm
import requests
import tarfile

import numpy as np
import xarray as xr
from scipy import ndimage


DEFAULT_DUMMY_LAT = 48.1411
DEFAULT_DUMMY_LON = 11.5777

DATA_DIR = Path(__file__).parent.parent / ".data"

download_files = {
    "d2c": {
        "url": "https://www.dropbox.com/s/1jhren60kmysr0w/dist2coast_1deg_merged.tar?dl=1",
        "size": 330857131,
    },
    "pysamosa_data": {
        "url": "https://www.dropbox.com/s/ygagj9gebnhsm5b/pysamosa_data.tar?dl=1",
        "size": 218451704,
    },
}


def download_dist2coast_nc():
    id = "d2c"
    url = download_files[id]["url"]
    fsize = download_files[id]["size"]
    fpath = DATA_DIR / Path(url).name.split("?")[0]
    print(fpath)

    # delete corrupt file (with wrong size!?)
    if fpath.exists() and fpath.stat().st_size != fsize:
        fpath.unlink()

    download_untar_file(url=url, dest_file=fpath, total_file_size=fsize)


def download_pysamosa_data():
    id = "pysamosa_data"
    url = download_files[id]["url"]
    fsize = download_files[id]["size"]
    fpath = DATA_DIR / Path(url).name.split("?")[0]

    # delete corrupt file (with wrong size!?)
    if fpath.exists() and fpath.stat().st_size != fsize:
        fpath.unlink()

    download_untar_file(url=url, dest_file=fpath, total_file_size=fsize)


def download_untar_file(url: str, dest_file: str, total_file_size: int = None):
    os.umask(000)
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_file, "wb") as f:
        with tqdm(
            total=total_file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest_file}... (Total {total_file_size / (1024**3):.2f}GB)",
            ascii=True,
        ) as pbar:
            for chunk in requests.get(url, stream=True).iter_content(32 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    # extract downloaded zip file
    with tarfile.open(dest_file) as tar:
        tar.extractall(path=dest_file.parent)


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
    id = "d2c"
    url = download_files[id]["url"]
    fsize = download_files[id]["size"]
    fpath = DATA_DIR / Path(url).name.split("?")[0]
    nc_file = fpath.parent / (fpath.stem + ".nc")

    # delete corrupt file (with wrong size!?)
    if fpath.exists() and fpath.stat().st_size != fsize:
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
        nc_file, engine="h5netcdf", chunks=pre_loaded_chunk_size
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
