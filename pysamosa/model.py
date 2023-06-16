from bisect import bisect_left
from itertools import chain

import numpy as np

from pysamosa.common_types import (
    ModelParameter,
    ModelSettings,
    SensorSettings,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.conf_params import CONST_A, CONST_B, CONST_C, CONST_F
from pysamosa.model_helpers import get_f_from_lut, load_samosa_luts
from pysamosa.rip import RipAnalyser


def get_region_max(wf, region_center, n_before_after):
    return np.max(wf[region_center - n_before_after : region_center + n_before_after])


def get_region_argmax(wf, region_center, n_before_after):
    return np.argmax(
        wf[region_center - n_before_after : region_center + n_before_after]
    ) + (region_center - n_before_after)


def get_closest_idxs(array, values):
    array = np.array(array)
    idxs = np.searchsorted(array, values, side="left")  # get insert positions

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return idxs


class SamosaModel:
    def __init__(
        self,
        model_sets: ModelSettings,
        sensor_sets: SensorSettings,
        wf_sets: WaveformSettings,
        settings_preset: SettingsPreset,
    ):
        self.sensor_sets = sensor_sets
        self.model_sets = model_sets
        self.wf_sets = wf_sets
        self.settings_preset = settings_preset

        self.B_r_Hz_oversampled = (
            self.sensor_sets.B_r_Hz
            * self.wf_sets.zp_oversampling_factor
            * self.wf_sets.internal_oversampling_factor
        )
        self.luts_samosa = load_samosa_luts()[self.sensor_sets.sensor_type.value]

    def _check_param(self, Pu: float, Hs: float, t0_ns: float):
        if not self.model_sets.Disable_LUT_Flag and not (
            self.luts_samosa["lut_alpha_p"][0][0]
            <= Hs
            <= self.luts_samosa["lut_alpha_p"][-1][0]
        ):
            raise RuntimeError(
                f"Hs must be within range [{self.luts_samosa['lut_alpha_p'][0][0]}, {self.luts_samosa['lut_alpha_p'][-1][0]}]"
            )

    def get_waveform_singlelook(
        self,
        Pu: float,
        Hs: float,
        t0_ns: float,
        fa_Hz: float,
        nu: float = 0.0,
        *,
        model_params,
        custom_gamma0=None,
    ):
        """
        :param Pu: waveform power amplitude
        :param Hs: significant wave height
        :param t0_ns: epoch relative to central gate in ns
        :param fa_Hz: Doppler frequency at which to calculate the single-look theoretical delay waveform
        :param model_params: ModelParameter() object with the model configuration
        :return:
        """

        self._check_param(Pu=Pu, Hs=Hs, t0_ns=t0_ns)

        # SL1. Initialize output variables to default values
        Np = int(self.wf_sets.np) - (self.wf_sets.internal_oversampling_factor - 1)
        h = model_params.alt_m

        # SL2. Compute local Earth radius, Re, and spherical surface parameter, alpha
        # Compute the local radii of curvature of the Earth's surface (Maulik Jain,
        # Improved sea level determination in the Arctic regions through development
        # of tolerant altimetry retracking, Eq. 5.2, pp. 47)
        # lat_geocentric = np.arctan((1 - CONST_F)**2 * np.tan(model_params.lat_rad))
        # Re = np.sqrt(CONST_A**2 * np.cos(lat_geocentric)**2 + CONST_B**2 *
        # np.sin(lat_geocentric)**2)  #more accurate (lat-dependent) calculus
        # of Earth' radius
        Re = np.sqrt(
            CONST_A**2 * np.cos(model_params.lat_rad) ** 2
            + CONST_B**2 * np.sin(model_params.lat_rad) ** 2
        )
        alpha = 1 + h / Re

        # SL3. Calculate model_paramsmodel_params.lat_radlat_rad new parameters
        prf_hz = (
            self.sensor_sets.prf_Hz
            if self.sensor_sets.prf_Hz is not None
            else 1 / model_params.pri_hz
        )
        # lambda0 = c / self.sensor_sets['f_c_Hz']
        dtau = 1 / self.B_r_Hz_oversampled
        dfa = prf_hz / self.sensor_sets.n_b
        xp = h * np.tan(model_params.ksix_rad)
        yp = -h * np.tan(model_params.ksiy_rad)
        alphax = (
            self.sensor_sets.sh_x
            * 8
            * np.log(2)
            / (h**2 * self.sensor_sets.theta_x_rad**2)
        )
        alphay = (
            self.sensor_sets.sh_y
            * 8
            * np.log(2)
            / (h**2 * self.sensor_sets.theta_y_rad**2)
        )
        Tb = self.sensor_sets.n_b / prf_hz
        Lx = (CONST_C * h) / (
            2 * model_params.Vs_m_per_s * self.sensor_sets.f_c_Hz * Tb
        )
        Ly = np.sqrt((CONST_C * h) / (alpha * self.sensor_sets.B_r_Hz))
        Lz = CONST_C / (2 * self.sensor_sets.B_r_Hz)
        Lg = alpha / (2 * h * alphay)
        sigmaz = Hs / 4

        # SL4. Define indices in delay (K) and doppler (L) space
        epoch_ref_gate = (
            model_params.epoch_ref_gate * self.wf_sets.internal_oversampling_factor
        )

        t_start_val = -epoch_ref_gate
        t = (np.arange(t_start_val, t_start_val + Np)) * dtau
        tau = t - float(t0_ns) * 1e-9
        K = tau * self.sensor_sets.B_r_Hz
        L = int(np.round(fa_Hz / dfa))

        # SL5. Define a value for alph_P
        if (
            self.settings_preset == SettingsPreset.SAMPLUS
            and 0.0 <= model_params.dist2coast < 10.0
        ):
            alpha_p = 0.55
        elif self.model_sets.Disable_LUT_Flag:
            alpha_p = self.model_sets.alpha_p_mean
        else:
            alpha_p = self.luts_samosa["lut_alpha_p"][
                bisect_left(self.luts_samosa["lut_alpha_p"][:, 0], Hs), 1
            ]

        # SL5. Calculate GL and GLK
        track_sign = -1 if model_params.ascending else 1
        orbit_slope = track_sign * (
            (CONST_A**2 - CONST_B**2) / (2 * Re**2)
        ) * np.sin(2 * model_params.lat_rad) - (
            -model_params.h_rate_m_per_s / model_params.Vs_m_per_s
        )
        Ls = orbit_slope * h / (alpha * Lx)
        # for S6: Enable_Slope_Effect_Flag is inherently turned on because a
        # different GL formula is used
        if self.model_sets.Enable_Slope_Effect_Flag:
            gamma = 2 * (L - Ls) * Lx**2 / Ly**2
        else:
            gamma = 2 * L * Lx**2 / Ly**2

        sign = Hs / np.abs(Hs) if Hs != 0 else 1

        # sigma_p = alpha_p / self.B_r_Hz_oversampled
        # theta_l = Lx / h * (L - Ls)
        # GL = 1 / (self.B_r_Hz_oversampled * np.sqrt(sigma_p**2 * (1 + (Lx /
        # Lz)**2 * (alpha * theta_l)**2) + (2 * sigmaz / CONST_C)**2))
        # #alternative formula from Dinardo et al. 2018
        GL = 1 / np.sqrt(
            alpha_p**2 + alpha_p**2 * gamma**2 + sign * (Hs / (4 * Lz)) ** 2
        )

        GLK = GL * K

        # SL6. Calculate GAMMA0 and Const
        XL = L * Lx
        YK = np.zeros(Np)
        YK[K > 0] = Ly * np.sqrt(K[K > 0])

        if (
            self.settings_preset == SettingsPreset.SAMPLUSPLUS
            and custom_gamma0 is not None
        ):
            gamma0 = custom_gamma0
        else:
            gamma0 = np.exp(
                -alphay * yp**2
                - alphax * (XL - xp) ** 2
                - XL**2 * nu / h**2
                - (alphay + nu / h**2) * YK**2
            ) * np.cosh(2 * alphay * yp * YK)
        # here it is intended to use the constant alpha_p term
        Const = self.model_sets.alpha_p_mean**2 * np.sqrt(2 * np.pi)

        # SL6. Calculate Tk
        Tk = np.zeros(Np)
        Tk[K > 0] = (1 + nu / (alphay * h**2)) - (
            yp / (Ly * np.sqrt(K[K > 0]))
        ) * np.tanh(2 * alphay * yp * Ly * np.sqrt(K[K > 0]))
        Tk[K <= 0] = (1 + nu / (alphay * h**2)) - 2 * alphay * yp**2

        # SL7. Calculate the final expression of Pr_SL
        f_0_l_k = get_f_from_lut(GLK, self.luts_samosa["lut_F0"], 0)
        f_1_l_k = get_f_from_lut(GLK, self.luts_samosa["lut_F1"], 1)

        if self.model_sets.Disable_SAM_F1_Flag:
            Pr_SL = Const * np.sqrt(GL) * gamma0 * f_0_l_k
        else:
            Pr_SL = (
                Const
                * np.sqrt(GL)
                * gamma0
                * (f_0_l_k + (sigmaz / Lg) * (sigmaz / Lz) * GL * Tk * f_1_l_k)
            )

        return Pr_SL

    def get_waveform_multilook(self, Pu, Hs, t0_ns, nu=0, *, model_params=None):
        model_params = model_params if model_params else ModelParameter()

        self._check_param(Pu=Pu, Hs=Hs, t0_ns=t0_ns)

        prf_hz = (
            self.sensor_sets.prf_Hz
            if self.sensor_sets.prf_Hz is not None
            else 1 / model_params.pri_hz
        )
        # ML1. Initialize output variables to default values

        # ML2. Calculate idealised Doppler beam angles if not available from
        # the SAR L1B input file
        h = model_params.alt_m
        Vs = model_params.Vs_m_per_s
        Tb = self.sensor_sets.n_b / prf_hz
        Np = int(self.wf_sets.np) - (self.wf_sets.internal_oversampling_factor - 1)

        # Compute the local radii of curvature of the Earth's surface (Maulik Jain,
        # Improved sea level determination in the Arctic regions through development
        # of tolerant altimetry retracking, Eq. 5.2, pp. 47)
        np.arctan((1 - CONST_F) ** 2 * np.tan(model_params.lat_rad))
        # Re = np.sqrt(CONST_A**2 * np.cos(lat_geocentric)**2 + CONST_B**2 * np.sin(lat_geocentric)**2)
        Re = np.sqrt(
            CONST_A**2 * np.cos(model_params.lat_rad) ** 2
            + CONST_B**2 * np.sin(model_params.lat_rad) ** 2
        )
        alpha = 1 + h / Re

        if self.model_sets.Ideal_Beam_Ang_Stack_flag or not len(
            model_params.beam_ang_stack_rad
        ):
            dTheta = (Vs * self.sensor_sets.bri) / (h * alpha)
            Theta1 = np.pi / 2 + dTheta * self.model_sets.Ideal_First_Look_Index_ML
            Theta2 = np.pi / 2 + dTheta * self.model_sets.Ideal_Last_Look_Index_ML
            self.Beam_Ang_Stack_TS = np.arange(Theta1, Theta2, dTheta)
        else:
            self.Beam_Ang_Stack_TS = model_params.beam_ang_stack_rad[
                ~np.isnan(model_params.beam_ang_stack_rad)
            ]

        # ML3. Calculate the vector of Doppler frequency of the Doppler beams
        # used for multi-looking
        lambda0 = CONST_C / self.sensor_sets.f_c_Hz
        Dopp_Freq_Stack_TS = (2 * Vs / lambda0) * np.cos(self.Beam_Ang_Stack_TS)

        # ML4. Sub-setting the Stack
        dfa = prf_hz / self.sensor_sets.n_b
        Beam_Indices_float = Dopp_Freq_Stack_TS / dfa
        factor = 1
        Beam_Index = (
            np.unique((np.round(Beam_Indices_float / factor)).astype(int)) * factor
        )
        Dopp_Freq_Stack_TS = Beam_Index * dfa

        # ML5. Build the Stack of Doppler beams prior to incoherent integration
        Neff = Dopp_Freq_Stack_TS.size
        self.Pr_Stack = np.zeros([Neff, Np])

        # Calculate GAMMA0
        custom_gamma0 = None
        if self.settings_preset == SettingsPreset.SAMPLUSPLUS:
            ripa = RipAnalyser(
                model_params.rip,
                sensor_sets=self.sensor_sets,
                model_params=model_params,
            )
            self.rip_params = ripa.rip_params

            custom_gamma0 = ripa.get_gamma0(n_looks_eff=Neff, n_gates=Np)

        for j, fa in enumerate(Dopp_Freq_Stack_TS):
            if self.model_sets.fit_zero_doppler:
                fa = 0.0

            if custom_gamma0 is not None:
                self.Pr_Stack[j] = self.get_waveform_singlelook(
                    Pu=Pu,
                    Hs=Hs,
                    t0_ns=t0_ns,
                    fa_Hz=fa,
                    nu=nu,
                    model_params=model_params,
                    custom_gamma0=custom_gamma0[:, j],
                )
            else:
                self.Pr_Stack[j] = self.get_waveform_singlelook(
                    Pu=Pu,
                    Hs=Hs,
                    t0_ns=t0_ns,
                    fa_Hz=fa,
                    nu=nu,
                    model_params=model_params,
                )

        # ML6. Mask out from the Stack of Doppler beams the power bins located
        # beyond the radar window length
        Lx = (lambda0 * h) / (2 * Vs * Tb)
        Beam_Range = h * (np.sqrt(1 + alpha * (Lx * Beam_Index / h) ** 2) - 1)
        dr = np.arange(0, Np)[::-1]
        Window_Range = CONST_C / (2 * self.B_r_Hz_oversampled) * dr

        if model_params.stack_mask_start_stop is not None:
            inds_stack_mask_mapped = get_closest_idxs(
                np.sort(Beam_Indices_float), np.sort(Beam_Index)
            )
            stack_mask_start_stop_mapped = model_params.stack_mask_start_stop[
                inds_stack_mask_mapped
            ].astype(int)

            for i, mask_val in np.ndenumerate(stack_mask_start_stop_mapped):
                self.Pr_Stack[i, mask_val:] = np.nan

        else:
            for j in np.arange(0, Neff):
                for i in np.arange(0, Np):
                    if Beam_Range[j] > Window_Range[i]:
                        self.Pr_Stack[j, i] = np.nan

        # ML7. Apply Weighting Function to the Stack prior to incoherent
        # summation
        Stack_Weights = np.ones(self.Pr_Stack.shape[0])
        self.Pr_Stack *= Stack_Weights[:, np.newaxis]

        # ML8. Calculate multi-looked waveform by incoherent summation across
        # the Stack
        self.Pr_ML = 1 / Neff * np.nansum(self.Pr_Stack, axis=0)
        # for S6: if avoid_zeros_in_the_multilooking is active then ignore the nan values
        # self.Pr_ML = np.nanmean(self.Pr_Stack, axis=0)
        # self.Pr_ML = np.nan_to_num(self.Pr_ML)

        self.model_max = np.max(self.Pr_ML)
        self.Pr_ML = Pu * (self.Pr_ML / self.model_max)

        # logging.debug(f'Generated multilooked waveform with params P_u={Pu:.4f}, Hs={Hs:.4f}, t0_ns={t0_ns:.4f}, nu={nu:.2e}')

        if self.wf_sets.internal_oversampling_factor > 1:
            self.Pr_ML = np.append(
                self.Pr_ML,
                np.repeat(
                    self.Pr_ML[-1], (self.wf_sets.internal_oversampling_factor - 1)
                ),
            )

        return self.Pr_ML

    def get_waveform_interference_mask(
        self,
        *,
        wf,
        t0_ns,
        model_params,
        fg_epoch_gates,
        n_normalise_grow,
        n_mask_grow,
        swh_max=None,
        mask_before_le=False,
    ):
        """Estimates the waveforms interference mask based on very-large-sea-state envelope.

        :param wf: the waveform to retrack
        :param t0_ns: the retracked epoch t0 in ns (referring to the epoch_ref_gate)
        :param model_params: the ModelParameters object for this measurement
        :param fg_epoch_gates: the estimated first-guess-epoch in gates
        :param n_normalise_grow: number of gates before and after a waveform's peak gate to normalise the waveform
        :param n_mask_grow: let the interference mask grow before and after the waveform gate that is larger than the envelope
        :param swh_max: maximum SWH boundary, from which the interference mask is generated
        :return: the envelope mask and the indices of the "bad" gates
        """
        if n_normalise_grow:
            wf = wf / get_region_max(
                wf, region_center=fg_epoch_gates, n_before_after=n_normalise_grow
            )
        else:
            wf = wf / np.max(wf)

        mask_max = self.get_waveform_singlelook(
            Pu=1.0, Hs=swh_max, t0_ns=t0_ns, nu=0.00, model_params=model_params, fa_Hz=0
        )
        mask_max = np.append(
            mask_max,
            np.repeat(mask_max[-1], self.wf_sets.internal_oversampling_factor - 1),
        )
        mask_max = mask_max / np.max(mask_max)
        interference_ref_waveform = mask_max

        mask_close_to_zero = np.isclose(interference_ref_waveform, 0.0, atol=1e-2)

        # set everything before max to 1.0
        mask_max_max = np.argmax(mask_max)
        interference_ref_waveform[:mask_max_max] = 1.0

        if mask_before_le:
            interference_ref_waveform[mask_close_to_zero] = 0.0

        # add safety margin to noise floor
        upper_noise_floor_bound = 0.05
        interference_ref_waveform += upper_noise_floor_bound

        if mask_before_le:
            interference_ref_waveform[mask_close_to_zero] = upper_noise_floor_bound

        # get indices of gates, for which the waveform is greater than the
        # modelled envelope
        inds_bad = (
            (wf.squeeze() > interference_ref_waveform.squeeze()).nonzero()[0].tolist()
        )

        # grow region of each bad ind by n_grow
        if len(inds_bad):
            if n_mask_grow:
                inds_bad = [
                    [i for i in range(j - n_mask_grow, j + n_mask_grow + 1)]
                    for j in inds_bad
                ]
                # flatten list and get unique set of bad inds
                inds_bad = sorted(list(set(list(chain.from_iterable(inds_bad)))))
            else:
                # flatten list and get unique set of bad inds
                inds_bad = list(set(inds_bad))

        # remove bad inds that are before fg_epoch_gates
        # inds_bad = [i for i in inds_bad if i >= fg_epoch_gates]

        # remove bad inds that are too close to the fg_epoch, meaning fg_epoch
        # +- n_mask_grow
        inds_bad = [i for i in inds_bad if (np.abs(fg_epoch_gates - i) > n_mask_grow)]
        inds_bad = [
            i for i in inds_bad if (np.abs(fg_epoch_gates - i) > n_normalise_grow)
        ]

        # remove invalid indices
        inds_bad = [i for i in inds_bad if (0 <= i < wf.shape[0])]

        return interference_ref_waveform, inds_bad
