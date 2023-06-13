import tempfile
from pathlib import Path

import cftime
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from bokeh.io import output_file, show
from bokeh.models import HoverTool
from bokeh.models.glyphs import Patches
from bokeh.plotting import ColumnDataSource, figure
from bokeh.tile_providers import Vendors

from pysamosa.data_access import _read_dataset_vars_from_ds
from pysamosa.settings import TEST_DATA_DIR

# data preparation
ncfile_list = sorted((TEST_DATA_DIR / "l1bs").rglob("*.nc"))

nc_filenames = [
    Path.cwd().parent
    / "data"
    / "s6"
    / "l2"
    / "S6A_P4_2__HR_STD__NT_038_018_20211120T051224_20211120T060836_F06.nc",
]

for nc_filename in nc_filenames:
    is_s6 = "S6A" in str(nc_filename)
    is_s3 = "S3" in str(nc_filename)
    grps = [
        g
        for g in list(netCDF4.Dataset(nc_filename).groups)
        if "data_20" in g or "data_140" in g
    ]
    nc_grp = grps[0] + "/ku" if len(grps) > 0 else ""

    nc_basename = (
        nc_filename.stem
        if "measurement" not in str(nc_filename)
        else nc_filename.parent.stem
    )

    if "dart" in str(nc_filename):
        cycle, ppass = int(nc_basename.split("_")[13]), int(nc_basename.split("_")[14])
    elif "_2_" in nc_basename:
        cycle, ppass = int(nc_basename.split("_")[8]), int(nc_basename.split("_")[9])
    elif "_1A_" in nc_basename:
        cycle, ppass = int(nc_basename.split("_")[13]), int(nc_basename.split("_")[14])
    elif "_1B_" in nc_basename:
        cycle, ppass = int(nc_basename.split("_")[13]), int(nc_basename.split("_")[14])
    else:
        cycle, ppass = 0, 0

    # level_str = (str(nc_filename.name).split('_')[2] if 'meas' not in str(nc_filename) else str(nc_filename.parent.name).split('_')[2]).lower()
    # # level_str = (str(nc_filename.name).split('_')[2] if 'meas' not in str(nc_filename) else str(nc_filename.name).split('_')[2]).lower()
    # level_str = level_str.replace('_1_', '_1b_')
    #
    # if level_str == '':
    #     level_str = '1a'
    #
    # is_l1a = f'_1a_' in str(nc_filename).lower()
    # is_l2 = f'_2_' in str(nc_filename).lower()

    with xr.open_dataset(nc_filename, decode_times=False, group=nc_grp) as dstmp:
        if is_s6:
            data_vars = {
                "lat_rad": [k for k in list(dstmp.coords.keys()) if "lat" in k][0],
                "lon_rad": [k for k in list(dstmp.coords.keys()) if "lon" in k][0],
                "time": [k for k in list(dstmp.coords.keys()) if "time" in k][0],
            }
        elif is_s3:
            data_vars = {
                "lat_rad": [
                    k for k in list(dstmp.coords.keys()) if "lat" in k and "sar" in k
                ][0],
                "lon_rad": [
                    k for k in list(dstmp.coords.keys()) if "lon" in k and "sar" in k
                ][0],
                "time": [
                    k for k in list(dstmp.coords.keys()) if "time" in k and "sar" in k
                ][0],
            }

    if "model" in str(nc_filename):
        data_vars["angle_of_approach_to_coast"] = "angle_of_approach_to_coast"

    ds = _read_dataset_vars_from_ds(
        nc_filename=nc_filename, data_var_names=data_vars, group=nc_grp
    )
    # nc_filename = LFSDATA_DIR / 'eumetsat/l1bs/S3A_SR_1_SRA_BS_20171210T145735_20171210T154803_20190705T205005_3028_025_239______MR1_R_NT_004.nc'
    # ds = _read_dataset_vars_from_ds(nc_filename=nc_filename, data_var_names=data_vars_eumetsat['l1b'])

    if isinstance(ds["time"][0], cftime.DatetimeGregorian):
        ds["time"] = ds["time"].astype("datetime64[ns]")

    df = pd.DataFrame(
        {
            k: v
            for k, v in ds.items()
            if k
            in [
                "time",
                "lat_rad",
                "lon_rad",
                "alt_m",
                "dist2coast",
                "distance_to_coast",
                "record_inds",
                "angle_of_approach_to_coast",
            ]
        }
    )

    # reduce dataset
    # df = df[::10]

    df["lat"] = np.degrees(df["lat_rad"])
    df["lon"] = np.degrees(df["lon_rad"])

    # df = df.drop(columns=['lat_rad', 'lon_rad'])

    # Define function to switch from lat/lon to mercator coordinates
    def x_coord(lat, lon):
        r_major = 6378137.000
        x = r_major * np.radians(lon)
        scale = x / lon
        y = (
            180.0
            / np.pi
            * np.log(np.tan(np.pi / 4.0 + lat * (np.pi / 180.0) / 2.0))
            * scale
        )
        return (x, y)

    # Define coord as tuple (lat,long)
    df["coordinates"] = list(zip(df["lat"], df["lon"]))

    # Create mercator column in our df
    df["mercator"] = [x_coord(lat, lon) for lat, lon in df["coordinates"]]

    # Split that column out into two separate columns - mercator_x and
    # mercator_y
    df[["mercator_x", "mercator_y"]] = df["mercator"].apply(pd.Series)

    # calc bearings (angles between points):
    def calc_bearings(lat, lon):
        """Calculate bearings between two vector of ellipsoidal coordinates.
        according to https://towardsdatascience.com/calculating-the-bearing-between-two-geospatial-coordinates-66203f57e4b4
        0.0 -> North, -pi/2 -> East, -pi -> South, -3/2 pi -> West

        returns bearings (in radians)
        """
        _lat = np.radians(np.asarray(lat))
        _lon = np.radians(np.asarray(lon))

        bearings = np.zeros(lat.size)
        for i in range(lat.size):
            if i != (lat.size - 1):
                lat0, lon0 = _lat[i], _lon[i]
                lat1, lon1 = _lat[i + 1], _lon[i + 1]
                dL = lon1 - lon0
                X = np.cos(lat1) * np.sin(dL)
                Y = np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(
                    dL
                )

                bearings[i] = np.arctan2(X, Y)
            else:
                bearings[i] = bearings[i - 1]

        bearings += np.pi  # so that it fits the definition
        return bearings

    bearings = calc_bearings(df["lat"], df["lon"])
    df["act_bearing"] = -bearings

    mercator_scale_factor = 1 / np.cos(df["lat_rad"])
    df["footprint_size_x"] = mercator_scale_factor * 10000
    df["footprint_size_y"] = mercator_scale_factor * 300

    # Plotting
    # palette = PRGn[11]  # Choose palette

    # Define color mapper - which column will define the colour of the data points
    # color_mapper = linear_cmap(field_name = 'AveragePrice', palette = palette, low = df['AveragePrice'].min(), high = df['AveragePrice'].max())

    # Set tooltips - these appear when we hover over a data point in our map,
    # very nifty and very useful
    tooltips = [
        ("latitude", "@lat"),
        ("longitude", "@lon"),
        ("dist2coast", "@dist2coast"),
        ("distance_to_coast", "@distance_to_coast"),
        ("altitude", "@alt_m"),
        ("record_ind", "@record_inds"),
        ("buoy_label", "@buoy_label"),
        ("angle_of_approach_to_coast", "@angle_of_approach_to_coast"),
    ]

    # Create the figure
    p = figure(
        title=f"{nc_basename}",
        # plot_width=1200, plot_height=900,
        sizing_mode="scale_width",
        aspect_ratio=2.0,
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_axis_label="longitude",
        y_axis_label="latitude",
        # tooltips=tooltips,
        tools="pan,wheel_zoom,box_zoom,save",
        toolbar_location="above",
        output_backend="webgl",
    )

    # Add map tile
    p.add_tile(Vendors.ESRI_IMAGERY)

    # Tell Bokeh to use df as the source of the data
    source = ColumnDataSource(data=df)
    # glyph = Rect(x="mercator_x", y="mercator_y", width='footprint_size_x', height='footprint_size_y', angle='act_bearing', fill_color='red', fill_alpha=0.6)
    # p.add_glyph(source, glyph)

    ####
    # hack because the hover area of Rect is shift (it is a bug:
    # https://github.com/bokeh/bokeh/issues/9752)
    rect_points_xy = np.array([-0.5, 0.5])
    xp = df["footprint_size_x"].values[:, np.newaxis] * rect_points_xy[np.newaxis]
    yp = df["footprint_size_y"].values[:, np.newaxis] * rect_points_xy[np.newaxis]

    def rotate(p, origin=(0, 0), angle_rad=0):
        R = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    xs = np.zeros((xp.shape[0], 4))
    ys = np.zeros((yp.shape[0], 4))
    for i in range(xp.shape[0]):
        x = np.array([xp[i, 0], xp[i, 0], xp[i, 1], xp[i, 1]])
        y = np.array([yp[i, 0], yp[i, 1], yp[i, 1], yp[i, 0]])

        # rotate points
        points_rotated = np.apply_along_axis(
            lambda p_i: rotate(p_i, angle_rad=-bearings[i]),
            axis=0,
            arr=np.vstack([x, y]),
        )
        xs[i, :] = points_rotated[0, :]
        ys[i, :] = points_rotated[1, :]

    xs = df["mercator_x"].values[:, np.newaxis] * np.ones((1, xs.shape[1])) + xs
    ys = df["mercator_y"].values[:, np.newaxis] * np.ones((1, ys.shape[1])) + ys

    source.add(xs.tolist(), "xs")
    source.add(ys.tolist(), "ys")

    glyph = Patches(
        xs="xs",
        ys="ys",
        fill_color="red",
        fill_alpha=0.6,
    )
    ####
    p.add_glyph(source, glyph)

    # Add a hover tool referring to the formatted columns
    hover = HoverTool(tooltips=tooltips)
    hover.point_policy = "follow_mouse"
    # hover.point_policy = 'snap_to_data'
    # hover.anchor = 'top_left'

    # Add the hover tool to the graph
    p.add_tools(hover)

    # Display in notebook
    output_file(
        Path(tempfile.gettempdir()) / "track_browser.html", title=f"{cycle}/{ppass}"
    )

    show(p)
