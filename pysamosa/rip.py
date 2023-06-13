import copy

import numpy as np
from astropy import modeling
from astropy.modeling import models
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

from pysamosa.common_types import ModelParameter, SensorSettings
from pysamosa.conf_params import CONST_C

# class RipAnalyserSettings(BaseModel):
#     model_params: ModelParameter
#     sensor_sets: SensorSettings


class RIPParameters(BaseModel):
    rip_az_fitted: np.ndarray = None
    nu: float = None
    mss: float = None
    pitch_mispoint_look: float = None
    pitch_mispoint_rad: float = None
    amplitude_fitted_norm: float = None
    fresnel_coeff_rf0: float = None
    halfpower_aperture_rad: float = None
    halfpower_looks: tuple = None

    class Config:
        arbitrary_types_allowed = True


def get_n_negative_looks(n_looks):
    return n_looks // 2 if n_looks % 2 else n_looks // 2 - 1


class RipAnalyser:
    def __init__(
        self,
        rip_wf: np.ndarray,
        sensor_sets: SensorSettings,
        model_params: ModelParameter,
    ):
        self.rip_wf = copy.deepcopy(rip_wf).reshape(
            [
                -1,
            ]
        )
        self.sensor_sets = sensor_sets
        self.model_params = model_params

        # oversample RIP before doing the fitting
        self.oversampling_factor = 8

        # normalize RIP
        self.rip_max = np.max(self.rip_wf)
        self.rip_wf_norm = self.rip_wf / self.rip_max

        self._rip_wf_len = self.rip_wf_norm.size
        self._n_negative_looks = get_n_negative_looks(self._rip_wf_len)
        self.doppler_beam_inds = np.arange(
            -self._n_negative_looks, self._rip_wf_len // 2 + 1
        )

        # calculate sensor-dependent params
        self.gamma_x = (
            self.sensor_sets.sh_x * 8 * np.log(2) / (self.sensor_sets.theta_x_rad**2)
        )
        Tb = self.sensor_sets.n_b / self.sensor_sets.prf_Hz
        self.Lx = (CONST_C * self.model_params.alt_m) / (
            2 * self.model_params.Vs_m_per_s * self.sensor_sets.f_c_Hz * Tb
        )
        self.dtheta_rad = self.Lx / self.model_params.alt_m
        self.theta_looks_rad = self.dtheta_rad * self.doppler_beam_inds

        # fit RIP
        self.rip_params = self._fit_along_track_rip_wf()

    def _fit_along_track_rip_wf(self):
        if self.oversampling_factor > 1.0:
            # oversample RIP
            f = interp1d(self.doppler_beam_inds, self.rip_wf_norm, kind="cubic")

            doppler_beam_inds = np.linspace(
                self.doppler_beam_inds[0],
                self.doppler_beam_inds[-1],
                num=self.oversampling_factor * self._rip_wf_len,
            )
            theta_looks_rad = np.linspace(
                self.theta_looks_rad[0],
                self.theta_looks_rad[-1],
                num=self.oversampling_factor * self._rip_wf_len,
            )
            self.rip_wf_oversampled = f(doppler_beam_inds)
        else:
            doppler_beam_inds = self.doppler_beam_inds
            theta_looks_rad = self.theta_looks_rad
            self.rip_wf_oversampled = self.rip_wf_norm

        # fit along-track RP using astropy.modeling.fitting module
        fitter = modeling.fitting.LevMarLSQFitter()
        model = models.Gaussian1D(amplitude=1.0, mean=0, stddev=1.0)
        self.rip_fitted_model = fitter(
            model, doppler_beam_inds, self.rip_wf_oversampled
        )

        # calc RIP params from fitted gaussian
        nu = (
            1 / (2 * (self.rip_fitted_model.stddev * self.dtheta_rad) ** 2)
            - self.gamma_x
        )
        mss = np.sqrt(1 / nu)
        fresnel_coeff_rf0 = mss * np.sqrt(2 * self.rip_max)

        self.rip_az_fitted = self.rip_fitted_model(doppler_beam_inds)

        # half-power aperture (depending on sea state, rough surface -> diffuse
        # RIP -> larger 3db aperture)
        n_inds_hpbw = argrelextrema(
            np.abs(self.rip_az_fitted - 0.5 * self.rip_fitted_model.amplitude.value),
            np.less,
        )[0]

        # check if gaussian fitted worked as expected
        if len(n_inds_hpbw) < 2:
            raise RuntimeError("RIP-Gaussian fitting failed. ")

        n_looks_hpbw = n_inds_hpbw[1] - n_inds_hpbw[0]
        hp_aperture_rad = n_looks_hpbw * self.dtheta_rad

        self.rip_params = RIPParameters(
            rip_az_fitted=self.rip_az_fitted[:: self.oversampling_factor],
            nu=nu,
            mss=mss,
            pitch_mispoint_look=self.rip_fitted_model.mean.value,
            pitch_mispoint_rad=np.interp(
                x=self.rip_fitted_model.mean.value,
                xp=doppler_beam_inds,
                fp=theta_looks_rad,
            ),
            amplitude_fitted_norm=self.rip_fitted_model.amplitude.value,
            fresnel_coeff_rf0=fresnel_coeff_rf0,
            halfpower_aperture_rad=hp_aperture_rad,
            halfpower_looks=tuple(
                (n_inds_hpbw / self.oversampling_factor - self._n_negative_looks)
            ),
        )

        return self.rip_params

    def get_gamma0(self, n_looks_eff, n_gates):
        """Calculates the gamma0 matrix/2D RIP. Interpolates the across-track RIP_act (which cannot be measured) from the along-track RIP_az

        :param n_looks_eff: number of effective looks that is used for the retracker (when generating the multilooked SAMOSA waveform model)
        :param n_gates: number of gates of the SAR power echo waveform
        :return: GAMMA0 matrix with shape (Np, Neff)
        """
        n_negative_looks_eff = get_n_negative_looks(n_looks_eff)
        self.looks_eff = np.arange(
            -n_negative_looks_eff, (n_looks_eff - n_negative_looks_eff)
        )

        # interpolate across-track RIP by taking the inner n_looks_eff effective looks
        # self.looks_rip_act = np.arange(0, (n_looks_eff - n_negative_looks_eff))
        self.looks_rip_act = np.linspace(
            0, (n_looks_eff - n_negative_looks_eff) - 1, num=n_gates
        )
        rip_act = self.rip_fitted_model(
            self.looks_rip_act + self.rip_fitted_model.mean.value
        )
        # f = interp1d(self.looks_rip_act, rip_act, kind='cubic')

        # self.gates = np.linspace(0, self.looks_rip_act[-1], num=n_gates)
        # self.rip_act_eff = f(self.gates)
        self.rip_act_eff = rip_act

        # calculate 2d RIP (along- and across-track)
        wf_start_look = get_n_negative_looks(self.rip_wf.size) - n_negative_looks_eff
        inds_act_eff = np.arange(wf_start_look, wf_start_look + n_looks_eff)
        self.gamma0_2d_rip = np.outer(
            self.rip_act_eff,
            self.rip_max
            * self.rip_az_fitted[:: self.oversampling_factor][inds_act_eff],
        )

        # normalise
        self.gamma0_2d_rip = self.gamma0_2d_rip / np.max(self.gamma0_2d_rip)

        return self.gamma0_2d_rip
