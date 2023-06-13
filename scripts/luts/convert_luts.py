import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pysamosa.common_types import SensorType


def read_f(name):
    LUT_X = []
    LUT_Y = []
    with open(name) as fid:
        for f in fid:
            if f[0] == "#":
                continue
            line = f.split()
            LUT_X.append(float(line[0]))
            LUT_Y.append(float(line[1]))

    LUT_X = np.array(LUT_X)
    LUT_Y = np.array(LUT_Y)
    return LUT_X, LUT_Y


def read_lut_alpha_p(filename_lut_alpha_p):
    SWH_lut, alpha_p_lut = [], []

    # filename_lut_alpha_p = 'LUT_Alpha_P_CS-2.txt'  # CS-2
    # filename_lut_alpha_p = 'alphap_table_SEN3_09_Nov_2017.txt'  # S3A

    with open(filename_lut_alpha_p) as fid:
        for li in fid:
            if li[0] == "#":
                continue
            line = li.split()
            SWH_lut.append(float(line[0]))
            alpha_p_lut.append(float(line[1]))

    return SWH_lut, alpha_p_lut


# S3 LUTs
LUT_X0, LUT_Y0 = read_f("F0.txt")
LUT_X1, LUT_Y1 = read_f("F1.txt")

F0 = np.transpose(np.array([LUT_X0, LUT_Y0]))
F1 = np.transpose(np.array([LUT_X1, LUT_Y1]))

lut_alpha_p_s3 = np.transpose(
    np.array(read_lut_alpha_p("alphap_table_SEN3_09_Nov_2017.txt"))
)

luts_s3 = {"lut_alpha_p": lut_alpha_p_s3, "lut_F0": F0, "lut_F1": F1}

# CS LUTs
LUT_X0, LUT_Y0 = read_f("F0.txt")
LUT_X1, LUT_Y1 = read_f("F1.txt")

F0 = np.transpose(np.array([LUT_X0, LUT_Y0]))
F1 = np.transpose(np.array([LUT_X1, LUT_Y1]))

lut_alpha_p_cs = np.transpose(np.array(read_lut_alpha_p("LUT_Alpha_P_CS-2.txt")))

luts_cs = {"lut_alpha_p": lut_alpha_p_cs, "lut_F0": F0, "lut_F1": F1}

# S6 F04 LUTs


def read_s6_lut(lut_file):
    with xr.open_dataset(lut_file) as ds:
        F0 = np.transpose(
            np.array([ds.LUT_F0_X.values, ds.LUT_F0_Y.values], dtype="float64")
        )
        F1 = np.transpose(
            np.array([ds.LUT_F1_X.values, ds.LUT_F1_Y.values], dtype="float64")
        )

        alpha_p_x_app = np.hstack(
            [np.linspace(-1, 0, 100, endpoint=False), ds.AlphaP_X.values]
        )
        alpha_p_y_app = np.hstack(
            [np.repeat(ds.AlphaP_Y.values[0], 100), ds.AlphaP_Y.values]
        )
        lut_alpha_p = np.transpose(np.array([alpha_p_x_app, alpha_p_y_app]))

        return {"lut_alpha_p": lut_alpha_p, "lut_F0": F0, "lut_F1": F1}


luts_s6_f04 = read_s6_lut("AUX_RLUT_S6A_002.nc")
luts_s6_f06 = read_s6_lut("AUX_RLUT_S6A_003.nc")
luts_s6_f06_ff = read_s6_lut(
    "S6A_TEST_AUX_FLUT___00000000T000000_99999999T999999_0001.NC"
)

all_luts = {
    SensorType.CS.value: luts_cs,
    SensorType.S3.value: luts_s3,
    SensorType.S6_F04.value: luts_s6_f04,
    SensorType.S6_F06.value: luts_s6_f06,
    SensorType.S6_F06_FF.value: luts_s6_f06_ff,
}

destfile_pickle = Path.cwd().parent.parent / "pysamosa" / "luts_samosa.pickle"
with open(destfile_pickle, "wb") as handle:
    pickle.dump(all_luts, handle)

# plot LUTS
for mission, luts in all_luts.items():
    plt.plot(luts["lut_alpha_p"][:, 0], luts["lut_alpha_p"][:, 1], label=mission)

plt.legend()
plt.grid()
plt.ylabel("alpha_p")
plt.xlabel("SWH [m]")
plt.show()

print(f"LUTs successfully written to {destfile_pickle}. ")
