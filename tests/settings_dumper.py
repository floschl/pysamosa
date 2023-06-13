import logging

# import imageio.v3 as iio
import imageio.v2 as iio
import numpy as np
from scipy.io import savemat


def dict_remove_None(d):
    return dict(filter(lambda item: item[1] is not None, dict(d).items()))


def replace_none(test_dict):
    # checking for dictionary and replacing if None
    if isinstance(test_dict, dict):
        for key in test_dict:
            if test_dict[key] is None:
                test_dict[key] = ""
            else:
                replace_none(test_dict[key])

    # checking for list, and testing for each value
    elif isinstance(test_dict, list):
        for val in test_dict:
            replace_none(val)


class SettingsDumper:
    def __init__(
        self, *, wf_sets, model_sets, sensor_sets, dest_mat_file, dest_gif_file
    ):
        self.dest_mat_file = dest_mat_file
        self.dest_gif_file = dest_gif_file
        self.l1b_nc_filename = None
        self.meas = []
        self.figs = []

        self.all_data = {
            "wf_sets": wf_sets.dict(),
            "model_sets": model_sets.dict(),
            "sensor_sets": sensor_sets.dict(),
            "meas": self.meas,
        }

    def add_retrack_entry(
        self, *, l1b_data_single, model_params, record_ind, res_fit, l2_data_single
    ):
        self.meas.append(
            {
                "record_ind": record_ind,
                "model_params": dict_remove_None(model_params),
                "l1b_data_single": dict_remove_None(l1b_data_single),
                "l2_data_single": l2_data_single,
                "res_fit": dict_remove_None(res_fit),
            }
        )

    def add_fig(self, f):
        self.figs.append(f)

    def write_out_mat(self):
        self.all_data["l1b_nc_filename"] = str(self.l1b_nc_filename)

        replace_none(self.all_data)

        savemat(str(self.dest_mat_file), self.all_data)
        logging.info(f"Writing retracked data to file {self.dest_mat_file}...")

    def export_gif(self):
        # save tmp pngs
        tmp_imgs = []
        for i, f in enumerate(self.figs):
            p = self.dest_gif_file.parent / f"tmp_img_{i}.png"
            f.savefig(p, dpi=200)
            tmp_imgs.append(p)

        if tmp_imgs:
            frames = np.stack([iio.imread(f) for f in tmp_imgs], axis=0)
            # iio.imwrite(self.dest_gif_file, frames, duration=0.2, loop=0)
            iio.mimsave(self.dest_gif_file, frames, duration=500, loop=0)
