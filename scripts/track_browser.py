from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models import HoverTool
from bokeh.models.glyphs import Circle
from bokeh.plotting import ColumnDataSource, figure
from bokeh.tile_providers import Vendors, get_provider

from pysamosa.data_access import _read_dataset_vars_from_ds

data_vars_eumetsat_s6 = {
    "time": "time",
    "lat_rad": "latitude",
    "lon_rad": "longitude",
}

# data_vars_eumetsat_s6 = {
#     'l1a': {
#         'time': 'time',
#         'lat_rad': 'latitude',
#         'lon_rad': 'longitude',
#     },
#     'l1b': {
#         'time': 'time',
#         'lat_rad': 'latitude',
#         'lon_rad': 'longitude',
#     },
#     'l2': {
#         'time': 'time',
#         'lat_rad': 'latitude',
#         'lon_rad': 'longitude',
#     },
# }

data_vars_eumetsat_cs = {
    "l1a": {
        "time": "time_20_ku",
        "lat_rad": "lat_20_ku",
        "lon_rad": "lon_20_ku",
    },
    "l1b": {
        "time": "time_20_ku",
        "lat_rad": "lat_20_ku",
        "lon_rad": "lon_20_ku",
    },
    "l2": {
        "time": "time_20_ku",
        "lat_rad": "lat_20_ku",
        "lon_rad": "lon_20_ku",
    },
}

data_vars_eumetsat_s3 = {
    "l1a": {
        "time": "time_l1a_echo_sar_ku",
        "lat_rad": "lat_l1a_echo_sar_ku",
        "lon_rad": "lon_l1a_echo_sar_ku",
    },
    "l1b": {
        "time": "time_l1b_echo_sar_ku",
        "lat_rad": "lat_l1b_echo_sar_ku",
        "lon_rad": "lon_l1b_echo_sar_ku",
    },
    "l2": {
        "time": "time_l1bs_echo_sar_ku",
        "lat_rad": "lat_l1bs_echo_sar_ku",
        "lon_rad": "lon_l1bs_echo_sar_ku",
    },
}

nc_filename = Path(
    "/nfs/DGFI145/C/work_flo/pysamosa_lfsdata/s3/l1b/S3A_SR_1_SRA____20180414T041038_20180414T050107_20190706T235251_3029_030_090______MR1_R_NT_004.nc"
)

level_str = (
    str(nc_filename.name).split("_")[2]
    if "meas" not in str(nc_filename)
    else str(nc_filename.parent.name).split("_")[2]
).lower()
level_str = level_str.replace("_1_", "_1b_")

if level_str == "":
    level_str = "1a"

is_l1a = "_1a_" in str(nc_filename).lower()
is_l2 = "_2_" in str(nc_filename).lower()
grps = [g for g in list(netCDF4.Dataset(nc_filename).groups) if "data_20" in g]
grp = f"{grps[0]}/ku" if len(grps) > 0 else ""

if "S6" in str(nc_filename):
    data_vars = data_vars_eumetsat_s6
elif "CS" in str(nc_filename):
    data_vars = data_vars_eumetsat_cs
else:
    data_vars = data_vars_eumetsat_s3["l2"] if is_l2 else data_vars_eumetsat_s3["l1b"]

ds = _read_dataset_vars_from_ds(
    nc_filename=nc_filename, data_var_names=data_vars, group=grp
)
# nc_filename = LFSDATA_DIR / 'eumetsat/l1bs/S3A_SR_1_SRA_BS_20171210T145735_20171210T154803_20190705T205005_3028_025_239______MR1_R_NT_004.nc'
# ds = _read_dataset_vars_from_ds(nc_filename=nc_filename, data_var_names=data_vars_eumetsat['l1b'])

df = pd.DataFrame(
    {
        k: v
        for k, v in ds.items()
        if k in ["time", "lat_rad", "lon_rad", "alt_m", "dist2coast", "record_inds"]
    }
)
df["lat"] = np.degrees(df["lat_rad"])
df["lon"] = np.degrees(df["lon_rad"])

# reduce data
if is_l1a:
    df = df[:: 140 // 2]
elif is_l2:
    # df = df[::20]
    df = df[::]
else:
    pass

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

# Split that column out into two separate columns - mercator_x and mercator_y
df[["mercator_x", "mercator_y"]] = df["mercator"].apply(pd.Series)

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
    ("record_ind", "@record_inds"),
]

# Create the figure
p = figure(
    title=f"{nc_filename.parent.name}/{nc_filename.stem}",
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
p.add_tile(get_provider(Vendors.ESRI_IMAGERY))

# Tell Bokeh to use df as the source of the data
df = df.drop(columns=["time"])
source = ColumnDataSource(data=df)

p.circle(
    x="mercator_x",
    y="mercator_y",
    # color=color_mapper,
    color="red",
    fill_alpha=0.9,
    source=source,
    # size=30,
    # radius=dist_lat_merc,
    radius=500,
)

# add Crete transponder
coords_crete_tr_deg = (35.3379, 23.7795)
coords_crete_tr = x_coord(*coords_crete_tr_deg)

source_tr = ColumnDataSource(
    {
        "x": [coords_crete_tr[0]],
        "y": [coords_crete_tr[1]],
        "radius": [700]
        # 'lat': coords_crete_tr_deg[0],
        # 'lon': coords_crete_tr_deg[1],
    }
)

glyph_circle_tr = Circle(
    x="x",
    y="y",
    radius="radius",
    radius_units="data",
    fill_color="yellow",
    fill_alpha=0.9,
    line_color="black",
)

p.add_glyph(source_tr, glyph_circle_tr)

# Add a hover tool referring to the formatted columns
hover = HoverTool(tooltips=tooltips)
hover.point_policy = "follow_mouse"
# hover.point_policy = 'snap_to_data'
# hover.anchor = 'top_left'

# Add the hover tool to the graph
p.add_tools(hover)

# Display in notebook
# output_notebook()
# output_file('track_browser.html')
# output_file()

# Show map
show(p)
