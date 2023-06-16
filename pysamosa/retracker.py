import logging

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import least_squares

from pysamosa.common_types import (
    FittingParameters,
    FittingSettings,
    ModelParameter,
    ModelSettings,
    RetrackerSettings,
    SensorSettings,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.leading_edge_detector import detect_leading_edge
from pysamosa.model import SamosaModel, get_region_argmax, get_region_max


def calc_misfit(wf1, wf2, inds_only=None, exclude_inds=None):
    if inds_only is not None and exclude_inds is not None:
        raise RuntimeError("Error calculating misfit. ")

    inds_only = np.asarray(inds_only) if inds_only is not None else None
    exclude_inds = np.asarray(exclude_inds) if exclude_inds is not None else None

    if exclude_inds is not None and exclude_inds.size:
        mask = np.ones(wf1.size, dtype=bool)
        mask[exclude_inds] = False
        misfit = 100 * np.sqrt(np.mean((wf1[mask] - wf2[mask]) ** 2))
    elif inds_only is not None and inds_only.size:
        # according to Dinardo, 2020, eq. 3.52
        misfit = 100 * np.sqrt(np.mean((wf1[inds_only] - wf2[inds_only]) ** 2))
    else:
        # according to Dinardo, 2020, eq. 3.52
        misfit = 100 * np.sqrt(np.mean((wf1 - wf2) ** 2))

    return misfit


def is_two_step_retracking_active_samplus(entropy, pulse_peakiness, misfit):
    """Checks for the four conditions as defined in Dinardo2020 Eq. 3.36

    :param entropy: entropy of the waveform as defined in Dinard2020 Eq. 3.37
    :param pulse_peakiness: pulse peakiness as defined in Dinardo2020 Eq. 3.38
    :param misfit: misfit between modelled and received waveform as defined in Dinardo2020 Eq. 3.52
    :returns: True if one of the conditions are met
    """
    return np.logical_or.reduce(
        (
            np.multiply(entropy, pulse_peakiness) < 0.68,
            np.multiply(entropy, pulse_peakiness) > 0.78,
            (100 * pulse_peakiness) > 4.0,
            (entropy / misfit) < 8.0,
        )
    )


def longest_consecutive_run(data):
    return sorted(
        np.split(data, np.where(np.diff(data) != 1)[0] + 1), key=len, reverse=True
    )[0]


def oversample_wf(wf, int_oversampling_fac):
    """Linearly upsamples waveform. Samples inbetween are interpolated.
    The very last (int_oversampling_fac - 1) are replicated from the very last interpolated sample

    :param wf: waveform to be oversampled
    :param int_oversampling_fac: internal oversampling factor, must be even-numbered
    :return: the upsampled waveform with length len(wf) * int_oversampling_fac
    """
    wf_len = len(wf)
    x = np.arange(wf_len)
    interpolator = Akima1DInterpolator(x, wf)
    # interpolator = PchipInterpolator(x, wf)
    x_interp = np.arange(
        0, wf_len - 1 + 1 / int_oversampling_fac, 1 / int_oversampling_fac
    )
    wf_oversampled = np.full(wf_len * int_oversampling_fac, np.nan)
    wf_oversampled[: -(int_oversampling_fac - 1)] = interpolator(x_interp)
    # replace last four values by the last valid ones
    wf_oversampled[-(int_oversampling_fac):] = wf_oversampled[-(int_oversampling_fac)]

    return wf_oversampled


class SamosaRetracker:
    def __init__(
        self,
        retrack_sets: RetrackerSettings,
        fitting_sets: FittingSettings,
        sensor_sets: SensorSettings,
        wf_sets: WaveformSettings,
    ):
        self.retrack_sets = retrack_sets
        self.fitting_sets = fitting_sets
        self.sensor_sets = sensor_sets
        self.wf_sets = wf_sets

        self.model_sets = ModelSettings.get_default_sets(
            st=sensor_sets.sensor_type,
            wf_sets=self.wf_sets,
        )

        if self.retrack_sets.fit_zero_doppler:
            self.model_sets.fit_zero_doppler = True

        # additional helper information
        self.B_r_Hz_oversampled = (
            self.sensor_sets.B_r_Hz
            * self.wf_sets.zp_oversampling_factor
            * self.wf_sets.internal_oversampling_factor
        )

    def _check_alt_flags(self):
        AO_Flag = True
        AM_Flag = True
        WQ_Flag = True

        return AO_Flag and AM_Flag and WQ_Flag

    def get_wf_max(wf, retrack_sets, fg_epoch=0):
        if retrack_sets.normalise_wf_by_fg_region:
            wf_max = get_region_max(
                wf, fg_epoch, retrack_sets.normalise_wf_by_fg_region
            )
        else:
            # Wf_in_Norm_part = wf[retrack_sets.Wf_Norm_First - 1:retrack_sets.Wf_Norm_Last]
            # wf_max = np.max(pd.Series(Wf_in_Norm_part).rolling(retrack_sets.Wf_Norm_Aw).mean().dropna().tolist())
            wf_max = np.max(wf)

        return wf_max

    def apply_fit_mask(*, wf, wf_sets, retrack_sets):
        os_factor = wf_sets.zp_oversampling_factor
        first_bin = (
            retrack_sets.Wf_Fit_First_Bin * os_factor
            if retrack_sets.Wf_Fit_First_Bin != -1
            else None
        )
        last_bin = (
            retrack_sets.Wf_Fit_Last_Bin * os_factor
            if retrack_sets.Wf_Fit_Last_Bin != -1
            else None
        )

        if first_bin:
            wf[:first_bin] = 0
        if last_bin:
            wf[last_bin + 1 :] = 0

        return wf

    def init_processing(self, l1b_data_single, *, disable_oversampling=False):
        Wf_in = l1b_data_single["wf"].squeeze()

        # perform an internal oversampling of the waveform to ease fitting
        # procedure
        int_oversampling_fac = self.wf_sets.internal_oversampling_factor
        if int_oversampling_fac > 1 and not disable_oversampling:
            Wf_in = oversample_wf(Wf_in, self.wf_sets.internal_oversampling_factor)

        if np.isnan(Wf_in).any():
            raise RuntimeError("Wf_in contains NaN values.")
        if (Wf_in.size) != self.wf_sets.np:
            raise RuntimeError(f"Wf_in must be of length Np(={self.wf_sets.np})")

        # Additional sanity checks on l1b_data_single
        if len(l1b_data_single["beam_ang_stack_rad"]) and np.all(
            np.isnan(l1b_data_single["beam_ang_stack_rad"])
        ):
            raise RuntimeError("Invalid beam_ang_stack_rad variable. ")

        self.dynamic_fg_epoch = None
        if "dynamic_fg_epoch" in l1b_data_single:
            self.dynamic_fg_epoch = (
                l1b_data_single["dynamic_fg_epoch"] * int_oversampling_fac
            )

        if (
            self.retrack_sets.normalise_wf_by_fg_region
            and "dynamic_fg_epoch" not in l1b_data_single
        ):
            raise RuntimeError("Invalid config. ")

        if "entropy" in l1b_data_single:
            self.entropy = l1b_data_single["entropy"]

        if "pulse_peakiness" in l1b_data_single:
            self.pulse_peakiness = l1b_data_single["pulse_peakiness"]

        # I1: Initialize output variables to default values
        # I2: Check altimeter flags to decide if conditions are met to proceed
        # with fitting
        if not self._check_alt_flags():
            raise RuntimeError("Wf_Fit_Proceed_Flag = False")

        # I3: Estimate the Thermal Noise
        os_factor = self.wf_sets.zp_oversampling_factor
        Wf_in_TN_part = Wf_in[
            (self.retrack_sets.Wf_TN_First * os_factor) : (
                self.retrack_sets.Wf_TN_Last * os_factor
            )
            + 2
        ]

        # in case of SAMPLUS-L1B waveforms, add 4.84 dB to thermal noise,
        # Dinardo2020 3.2.3
        if (
            (
                self.retrack_sets.settings_preset == SettingsPreset.SAMPLUS
                or self.retrack_sets.settings_preset == SettingsPreset.SAMPLUSPLUS
            )
            and "thermal_noise" in l1b_data_single
            and not np.isnan(l1b_data_single["thermal_noise"])
        ):
            self.TN = l1b_data_single["thermal_noise"]
        else:
            self.TN = np.mean(Wf_in_TN_part)

        # I4: Compute the waveform  maximum value to be used for normalisation
        if self.retrack_sets.Wf_Norm_Flag:
            if self.dynamic_fg_epoch:
                self.Wf_max = SamosaRetracker.get_wf_max(
                    Wf_in, self.retrack_sets, fg_epoch=self.dynamic_fg_epoch
                )
            else:
                self.Wf_max = SamosaRetracker.get_wf_max(Wf_in, self.retrack_sets)
            # The value of Wf_max needs to be checked against TN to determine
            # if normalisation can proceed.
            if self.Wf_max <= self.TN * self.retrack_sets.Thr:
                self.Wf_norm = Wf_in
                raise RuntimeError("Wf_max <= TN * Thr")

            # I5: Apply normalisation to the waveform
            self.Wf_norm = Wf_in / self.Wf_max
            self.TN = self.TN / self.Wf_max
        else:
            self.Wf_norm = Wf_in
            self.Wf_max = 1.0

        # adjust Wf_Fit_First_Bin/Wf_Fit_Last_Bin according to RAW or RMC mode
        if (
            "s6" in self.sensor_sets.sensor_type.value
            and self.retrack_sets.auto_detect_s6_rmc
        ):
            is_rmc_active = np.mean(self.Wf_norm[-self.wf_sets.np // 3 :]) < 1e-3

            if is_rmc_active:
                self.retrack_sets.Wf_Fit_First_Bin = 11
                self.retrack_sets.Wf_Fit_Last_Bin = 132
            else:
                self.retrack_sets.Wf_Fit_First_Bin = 1
                self.retrack_sets.Wf_Fit_Last_Bin = self.wf_sets.np

        if np.allclose(self.Wf_norm, 0.0):
            raise RuntimeError("Wf_in are all-zero after normalisation. ")

        # I6: Check user control flag Wf_Fit_Activate_Flag to see if the user wants to proceed with waveform fitting
        # if self.sets.Wf_Fit_Activate_Flag:

    def _get_least_square_weights(self):
        wf = self.Wf_norm
        ls_weights = np.ones(wf.shape[0])
        le_inds = None

        # leading edge weighting factor
        try:
            le_inds = detect_leading_edge(
                wf,
                fg_epoch=self.dynamic_fg_epoch,
                normalise_wf_by_fg_region=self.retrack_sets.normalise_wf_by_fg_region,
            )
            ls_weights[le_inds] *= self.retrack_sets.leading_edge_weight_factor
        except Exception as e:
            logging.debug(f"Error detecting leading edge: {e}")

        return ls_weights, le_inds

    def _fun_model(p, *args):
        kwargs = args[0]
        _self = kwargs["self"]
        wf = _self.Wf_norm
        retrack_sets = _self.retrack_sets
        model_params = kwargs["model_params"]
        fitting_params = kwargs["fitting_params"]
        ls_weights = kwargs["ls_weights"]

        epoch_init = p[0]

        logging.debug(
            f"Generating SAMOSA waveform with params: P_u={p[2]}, Hs={p[1]}, t0_ns={epoch_init}"
        )

        # return diff wf_model - wf if requested
        do_return_diff = (
            True if "return_diff" in kwargs and kwargs["return_diff"] else False
        )

        sm = SamosaModel(
            model_sets=_self.model_sets,
            sensor_sets=_self.sensor_sets,
            wf_sets=_self.wf_sets,
            settings_preset=retrack_sets.settings_preset,
        )

        # FIT5. Determine the theoretical model waveform to be used for the
        # fitting of the waveform
        try:
            if fitting_params.fit_nu_instead_of_swh:
                if retrack_sets.Disable_ML_Flag:
                    wf_model = sm.get_waveform_singlelook(
                        Pu=p[2],
                        Hs=fitting_params.swh_first_step,
                        t0_ns=epoch_init,
                        fa_Hz=0,
                        nu=p[1],
                        model_params=model_params,
                    )
                else:
                    wf_model = sm.get_waveform_multilook(
                        Pu=p[2],
                        Hs=fitting_params.swh_first_step,
                        t0_ns=epoch_init,
                        nu=p[1],
                        model_params=model_params,
                    )
            else:
                if retrack_sets.Disable_ML_Flag:
                    wf_model = sm.get_waveform_singlelook(
                        Pu=p[2],
                        Hs=p[1],
                        t0_ns=epoch_init,
                        fa_Hz=0,
                        model_params=model_params,
                    )
                else:
                    wf_model = sm.get_waveform_multilook(
                        Pu=p[2], Hs=p[1], t0_ns=epoch_init, model_params=model_params
                    )
        except RuntimeWarning as e:
            logging.warning(
                f'Warning in generating multilooked waveform: {" ".join(e.args)}'
            )
            wf_model = np.zeros(_self.wf_sets.np)

        if np.allclose(wf_model, 0.0):
            logging.warning(
                f"Calculated multilooked waveform is zero. Params: P_u={p[2]}, Hs={p[1]}, t0_ns={epoch_init}"
            )
            _self.res_fit_info["wf_opt"] = wf_model
            return wf_model

        wf_model += _self.TN
        diff = wf_model - wf

        diff *= ls_weights

        if (
            retrack_sets.interference_masking
            and not fitting_params.disable_interference_masking
        ):
            sm.wf_sets = _self.wf_sets  # use oversampled model waveform in this case

            interference_mask, interference_inds = sm.get_waveform_interference_mask(
                wf=wf,
                t0_ns=epoch_init,
                model_params=model_params,
                fg_epoch_gates=_self.dynamic_fg_epoch,
                n_normalise_grow=retrack_sets.normalise_wf_by_fg_region,
                n_mask_grow=retrack_sets.interference_masking_grow,
                swh_max=retrack_sets.interference_masking_swh_max,
                mask_before_le=retrack_sets.interference_masking_mask_before_le,
            )

            # remove overweighted indices from ls_weights
            le_inds = (ls_weights > 1.0).nonzero()[0]
            interference_inds = [ei for ei in interference_inds if ei not in le_inds]

            diff[interference_inds] = 0.0
            _self.res_fit_info["interference_inds"] = interference_inds
            _self.res_fit_info["interference_mask"] = interference_mask

        _self.res_fit_info["wf_opt"] = wf_model
        _self.res_fit_info["wf_model_max"] = sm.model_max

        # subwaveform approach
        if _self.retrack_sets.subwaveform_mode:
            max_le_gate = get_region_argmax(
                wf,
                region_center=_self.dynamic_fg_epoch,
                n_before_after=_self.retrack_sets.normalise_wf_by_fg_region,
            )
            diff[max_le_gate + _self.retrack_sets.subwaveform_n_gates_after_le :] = 0.0

            _self.res_fit_info["max_le_gate"] = max_le_gate

        res = diff if do_return_diff else wf_model
        res = SamosaRetracker.apply_fit_mask(
            wf=res, wf_sets=_self.wf_sets, retrack_sets=_self.retrack_sets
        )

        return res

    def _run_fitting_process(
        self, model_params: ModelParameter, fitting_params: FittingParameters = None
    ):
        fitting_params = (
            fitting_params if fitting_params is not None else FittingParameters()
        )

        fg_gate = (
            self.dynamic_fg_epoch if self.dynamic_fg_epoch else np.argmax(self.Wf_norm)
        )

        # FIT4. Determine the variables to be fitted, the number of variables to be fitted and their initial values
        # Vars to be fitted: [t0, Hs, Pu]

        # epoch/t0
        minmax_epoch = (
            fitting_params.minmax_epoch
            if fitting_params.minmax_epoch
            else self.fitting_sets.Fit_Var_1_MinMax_epoch
        )
        init_epoch = (
            fitting_params.init_epoch[0]
            if fitting_params.init_epoch
            else self.t[fg_gate] / 1e-9
        )

        # Hs
        if fitting_params.fit_nu_instead_of_swh:
            minmax_hs_or_nu = self.fitting_sets.Fit_Var_5_MinMax_nu
            init_hs_or_nu = self.fitting_sets.Fit_Var_5_Init_nu
        else:
            minmax_hs_or_nu = (
                fitting_params.minmax_swh
                if fitting_params.minmax_swh
                else self.fitting_sets.Fit_Var_2_MinMax_Hs
            )
            init_hs_or_nu = (
                fitting_params.init_swh
                if fitting_params.init_swh
                else self.fitting_sets.Fit_Var_2_Init_Hs
            )

        # Pu
        minmax_pu = (
            fitting_params.minmax_pu
            if fitting_params.minmax_pu
            else self.fitting_sets.Fit_Var_3_MinMax_Pu
        )
        init_pu = (
            fitting_params.init_pu
            if fitting_params.init_pu
            else self.fitting_sets.Fit_Var_3_Init_Pu
        )

        Fit_Var_Init = np.array([init_epoch, init_hs_or_nu, init_pu])

        # FIT5. Determine the theoretical model waveform to be used for the fitting of the waveform
        # FIT6. Computation of the new estimates by non-linear least-square fit
        ls_weights, le_inds = self._get_least_square_weights()

        _fun_model_arg = (
            {
                "self": self,
                "model_params": model_params,
                "fitting_params": fitting_params,
                "ls_weights": ls_weights,
                "subwaveform_stopgate": 0,
            },
        )

        self.res_fit_info = {
            "interference_inds": np.array([]),
            "interference_mask": np.ones(self.Wf_norm.shape[0]),
        }

        logging.debug(
            f"Started fitting. Init values: swh={init_hs_or_nu:.2f}m,"
            f"epoch={init_epoch:.2f}ns, Pu={init_pu:.2f}W"
        )

        ns_per_gate = self.dtau * 1e9
        epoch_min, epoch_max = minmax_epoch[0], minmax_epoch[1]

        if self.dynamic_fg_epoch and self.fitting_sets.limit_epoch_around_fg_n:
            n_gates_before_after = self.fitting_sets.limit_epoch_around_fg_n
            epoch_min = init_epoch - n_gates_before_after * ns_per_gate
            epoch_max = init_epoch + n_gates_before_after * ns_per_gate

        Fit_Var_Bounds = (
            [epoch_min, minmax_hs_or_nu[0], minmax_pu[0]],
            [epoch_max, minmax_hs_or_nu[1], minmax_pu[1]],
        )

        max_nfev = self.fitting_sets.Fit_MaxIter

        res_ls = least_squares(
            fun=SamosaRetracker._fun_model,
            x0=Fit_Var_Init,
            bounds=Fit_Var_Bounds,
            method="trf",
            args=({**_fun_model_arg[0], **{"return_diff": True}},),
            ftol=self.fitting_sets.trf_threshold,
            gtol=self.fitting_sets.trf_threshold,
            xtol=self.fitting_sets.trf_threshold,
            diff_step=self.fitting_sets.trf_stepsize,
            max_nfev=max_nfev,
            # verbose=2,
            # **loss_kw,
        )

        if not res_ls.success:
            RuntimeError(f"Failed TRF fitting: {res_ls.message}. ")

        if fitting_params.fit_nu_instead_of_swh:
            res_fit = {
                "swh": fitting_params.swh_first_step,
                "epoch_ns": res_ls.x[0],
                "Pu": res_ls.x[2],
                "nu": res_ls.x[1],
                "n_iter": res_ls.nfev,
                **self.res_fit_info,
            }
        else:
            res_fit = {
                "swh": res_ls.x[1],
                "epoch_ns": res_ls.x[0],
                "Pu": res_ls.x[2],
                "nu": 0,
                "n_iter": res_ls.nfev,
                **self.res_fit_info,
            }

        self.res_fit_info["wf_opt"] = SamosaRetracker.apply_fit_mask(
            wf=self.res_fit_info["wf_opt"],
            wf_sets=self.wf_sets,
            retrack_sets=self.retrack_sets,
        )

        # take fitted waveform that was generated at the very last step
        wf_opt = (
            self.res_fit_info["wf_opt"]
            if "wf_opt" in self.res_fit_info
            else np.zeros(len(self.Wf_norm))
        )

        # FIT7: Assign fitted variables to the output values
        # (FIT8. Re-convert Epoch in seconds)
        # FIT9. De-normalise the retrieved Power and Noise Values
        factor_denorm = self.Wf_max / self.res_fit_info["wf_model_max"]

        misfit = calc_misfit(wf_opt, self.Wf_norm)
        if self.retrack_sets.subwaveform_mode:
            misfit_selective = calc_misfit(
                wf_opt[
                    : res_fit["max_le_gate"]
                    + self.retrack_sets.subwaveform_n_gates_after_le
                ],
                self.Wf_norm[
                    : res_fit["max_le_gate"]
                    + self.retrack_sets.subwaveform_n_gates_after_le
                ],
            )
        else:
            misfit_selective = calc_misfit(
                wf_opt,
                self.Wf_norm,
                exclude_inds=self.res_fit_info["interference_inds"],
            )

        return {
            **res_fit,
            **{
                "wf": self.Wf_norm,
                "wf_opt": wf_opt,
                "Pu_denorm": res_fit["Pu"] * factor_denorm,
                "factor_denorm": factor_denorm,
                "misfit": misfit,
                "misfit_le": calc_misfit(wf_opt, self.Wf_norm, inds_only=le_inds),
                "misfit_selective": misfit_selective,
                "le_inds": le_inds,
                "l2norm": np.linalg.norm(self.Wf_norm - wf_opt),
            },
        }

    def fit_wf(self, l1b_data_single, model_params: ModelParameter):
        self.init_processing(l1b_data_single)

        # FIT1. Initialize output variables to default values
        # FIT2. Determine if waveform fitting should proceed
        # FIT3. Calculate the waveform time abscissa
        self.dtau = 1 / self.B_r_Hz_oversampled
        epoch_ref_gate = (
            model_params.epoch_ref_gate * self.wf_sets.internal_oversampling_factor
        )
        self.t = (
            np.arange(-epoch_ref_gate, -epoch_ref_gate + self.wf_sets.np) * self.dtau
        )

        # FIT4-9
        res_fit = self._run_fitting_process(model_params)

        if self.retrack_sets.interference_masking_second_retracking_step:
            n_iter_first = res_fit["n_iter"]

            logging.debug(
                f'Starting second iteration step with updated interference_masking_swh_max={res_fit["swh"]}. '
            )
            upper_swh_margin = 2.0
            self.retrack_sets.interference_masking_swh_max = np.min(
                [
                    res_fit["swh"] + upper_swh_margin,
                    self.fitting_sets.Fit_Var_2_MinMax_Hs[1],
                ]
            )
            res_fit = self._run_fitting_process(model_params)

            res_fit["n_iter"] = (n_iter_first, res_fit["n_iter"])

        # SAMOSA+ addition: second retracking step
        if self.retrack_sets.second_retracking_step_samplus:
            if (
                "entropy" in dir(self)
                and "pulse_peakiness" in dir(self)
                and is_two_step_retracking_active_samplus(
                    entropy=self.entropy,
                    pulse_peakiness=self.pulse_peakiness,
                    misfit=res_fit["misfit"],
                )
            ):
                n_iter_first = res_fit["n_iter"]

                fitting_params = FittingParameters(
                    fit_nu_instead_of_swh=True,
                    swh_first_step=res_fit["swh"],
                    disable_interference_masking=False,
                )
                logging.debug(
                    f'Starting second iteration step, taking SWH={res_fit["swh"]} from first step. '
                )
                res_fit = self._run_fitting_process(
                    model_params=model_params, fitting_params=fitting_params
                )

                res_fit["n_iter"] = (n_iter_first, res_fit["n_iter"])

        logging.debug(
            f'Finished fitting (n_iter={res_fit["n_iter"]}). Found params: swh={res_fit["swh"]:.2f}m, '
            f'epoch={res_fit["epoch_ns"]:.2f}ns, Pu={res_fit["Pu"]:.2f}W, nu={res_fit["nu"]:.2e}. '
        )

        return res_fit
