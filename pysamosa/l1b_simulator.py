import numpy as np
from numpy.random import default_rng

from pysamosa import simple_logger
from pysamosa.common_types import ModelSettings, SensorSettings, SettingsPreset
from pysamosa.conf_params import CONST_C
from pysamosa.data_access import get_model_param_obj_from_l1b_data
from pysamosa.model import SamosaModel

# fixed values taken from ind 42660, file
# RES_S3A_SR_1_SRA_A__20200108T101712_20200108T110742_20200202T201001_3029_053_279_GPOD_SAR_O_NT_003
l1b_data_single_template = {
    "wf": None,
    "lat_rad": -0.5918622555846702,
    "lon_rad": -0.3281458093122721,
    "alt_m": 815770.4278019192,
    "h_rate_m_per_s": 24.03475190775509,
    "Vs_m_per_s": 7534.799030740116,
    "ksix_rad": 2.094395102393196e-05,
    "ksiy_rad": 3.490658503988659e-06,
    "epoch_ref_gate": 64.0,
    "beam_ang_stack_rad": np.array([]),  # take ideal angles
    "dist2coast": 668,
}


class L1bSimulator:
    def __init__(
        self,
        *,
        model_sets: ModelSettings,
        swh,
        Pu=1.0,
        sensor_sets=None,
        wf_sets,
        settings_preset,
        add_thermal_speckle_noise=True,
        add_interference=False
    ):
        self.l1b_data_single = l1b_data_single_template

        self.model_sets = model_sets
        self.model_params = get_model_param_obj_from_l1b_data(
            l1b_data_single_template, 0
        )
        self.sensor_sets = sensor_sets if sensor_sets is not None else SensorSettings()
        self.wf_sets = wf_sets
        self.settings_preset = settings_preset
        self.wf_len = self.wf_sets.np
        self.oversampling_factor = self.wf_sets.zp_oversampling_factor

        self.swh = swh
        self.dtau = 1 / (self.sensor_sets.B_r_Hz * self.wf_sets.zp_oversampling_factor)
        epoch_refgate = self.l1b_data_single["epoch_ref_gate"]
        retrack_point_gates = 38 if settings_preset == SettingsPreset.NONE else 310
        self.epoch_ns = ((retrack_point_gates - epoch_refgate) * self.dtau) * 1e9
        self.Pu = Pu

        self.sam_model = SamosaModel(
            model_sets=self.model_sets,
            sensor_sets=self.sensor_sets,
            wf_sets=wf_sets,
            settings_preset=settings_preset,
        )

        # noise
        self.rng = default_rng(42)  # generate noise generator object

        self.add_interference = add_interference
        self.add_thermal_speckle_noise = add_thermal_speckle_noise

        simple_logger.set_root_logger()

    def get_l2_data_single(self):
        return {
            "time": None,
            "lat_rad": None,
            "lon_rad": None,
            "swh": self.swh,
            "swh_qual": False,
            "Pu_W": self.Pu,
            "epoch_ns": self.epoch_ns,
            "range": (float(self.epoch_ns)) * 1e-9 * (CONST_C / 2),
            "tracker_range_m": 0.0,
            "alt_m": l1b_data_single_template["alt_m"],
            "n_iter": 0,
            "misfit": 0.0,
        }

    def get_awgn(self):
        # generate additive white gaussian noise (AWGN)
        mean, variance = 5e-3, 5e-4
        return mean + variance * self.rng.standard_normal(self.wf_len)

    def get_multi_noise(self):
        k = 200
        # k = 500
        # mult_noise = self.rng.gamma(shape=k, scale=0.5 * 1/k, size=self.wf_len)
        mult_noise = self.rng.gamma(shape=k, scale=0.4, size=self.wf_len)
        return mult_noise

    def get_interference_inds(self, wf):
        # interference = np.zeros(self.wf_len)

        wf_maxgate = np.argmax(wf)
        diff_start_gate_from_max = 15 * self.oversampling_factor

        rand_start_gate = (
            int(self.rng.uniform(-5, 5)) * self.oversampling_factor
        )  # randomised start_gate
        start_gate = wf_maxgate + diff_start_gate_from_max + rand_start_gate
        # randomised width of interference
        rand_dist_width = int(self.rng.uniform(10, 20)) * self.oversampling_factor

        # interference[start_gate:start_gate+rand_dist_width] = 1.0
        interference_mask = np.zeros(self.wf_len, dtype=bool)
        interference_mask[start_gate : start_gate + rand_dist_width] = True

        return interference_mask

    def __iter__(self):
        return self

    def __next__(self):
        wf = self.sam_model.get_waveform_multilook(
            Pu=self.Pu,
            Hs=self.swh,
            t0_ns=self.epoch_ns,
            nu=0,
            model_params=self.model_params,
        )
        wf = wf / np.max(wf)  # normalise waveform

        # add additive (thermal noise) and multiplicative (speckle) noise
        if self.add_thermal_speckle_noise:
            wf += self.get_awgn()
            wf *= self.get_multi_noise()

        wf = wf / np.max(wf)  # normalise waveform

        wf_argmax = int(np.argmax(wf))

        if self.add_interference:
            distorted_mask = self.get_interference_inds(wf=wf)
            # set distorted gates to a maximum waveform value of 1.0 (omit the
            # noise in these gates)
            wf[distorted_mask] = 1.0
            wf_argmax = int(np.argmax(wf[~distorted_mask]))

        return {
            **self.l1b_data_single,
            **{
                "wf": wf,
                "dynamic_fg_epoch": wf_argmax,
            },
        }
