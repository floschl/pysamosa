from pathlib import Path

import numpy as np
import xarray as xr

ncfile_l1b = Path(
    "/nfs/DGFI145/C/work_flo/cs2_files_samplus_test/CS2_open_ocean/CS_LTA__SIR_SAR_1B_20150503T160800_20150503T161329_D001.nc"
)
ncfile_l2gop = Path(
    "/nfs/DGFI145/C/work_flo/cs2_files_samplus_test/CS2_open_ocean/CS_LTA__SIR_GOP_2__20150503T155002_20150503T163939_C001_0269.nc"
)

l1b = xr.open_dataset(ncfile_l1b)
l2 = xr.open_dataset(ncfile_l2gop)

mask_latminmax = (l2.lat_20_ku >= (l1b.lat_20_ku.min() - 0.0001)) & (
    l2.lat_20_ku <= l1b.lat_20_ku.max()
)
l2_realigned = {
    k: v[mask_latminmax].values for k, v in dict(l2).items() if "time_20_ku" in v.dims
}

for k, v in l2_realigned.items():
    try:
        new_da = xr.DataArray(
            np.array(v.data),
            dims=["time_20_ku"],
            coords=dict(
                time_20_ku=l1b.coords["time_20_ku"].values,
                lat_20_ku=(["time_20_ku"], l1b.coords["lat_20_ku"].values),
                lon_20_ku=(["time_20_ku"], l1b.coords["lon_20_ku"].values),
            ),
        )
    except BaseException:
        pass
    l2_realigned[k] = new_da

l2_realigned["window_del_20_ku"] = xr.DataArray(
    l1b.window_del_20_ku.values,
    dims=["time_20_ku"],
    coords=dict(
        time_20_ku=l1b.coords["time_20_ku"].values,
        lat_20_ku=(["time_20_ku"], l1b.coords["lat_20_ku"].values),
        lon_20_ku=(["time_20_ku"], l1b.coords["lon_20_ku"].values),
    ),
)

ds_l2_realigned = xr.Dataset(l2_realigned)

ncfile_l2_dest = ncfile_l1b.parent / (
    str(ncfile_l1b.stem).replace("1B", "2_") + "_realigned" + ".nc"
)
ds_l2_realigned.to_netcdf(ncfile_l2_dest)
# mask_latminmax = l1b.lat
