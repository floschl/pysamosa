import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, model_validator


class SensorType(Enum):
    S3 = "s3"
    CS = "cs"
    S6_F04 = "s6_f04"
    S6_F06 = "s6_f06"
    S6_F04_FF = "s6_f04_ff"
    S6_F06_FF = "s6_f06_ff"


class L1bSourceType(Enum):
    GPOD = "gpod"
    EUM_S3 = "eum_s3"
    EUM_CS = "eum_cs"
    EUM_S6_F04 = "eum_s6_f04"
    EUM_S6_F06 = "eum_s6_f06"
    EUM_S6_F04_FFSAR = "eum_s6_ffsar_f04"
    EUM_S6_F06_FFSAR = "eum_s6_ffsar_f06"


class SettingsPreset(Enum):
    NONE = "none"  # take default preset for standard SAMOSA
    CORALv1 = "coralv1"
    CORALv2 = "coralv2"
    SAMPLUS = "samplus"
    SAMPLUSPLUS = "samplusplus"


class SensorSettings(BaseModel):
    sensor_type: SensorType = SensorType.S3
    f_c_Hz: float = 13.575e9  # central frequency [Hz]
    B_t_Hz: float = 320e6  # chirp bandwidth [Hz]
    B_r_Hz: float = 320e6  # receiver bandwidth [Hz]
    # 3dB beamwidth along track [rad], from S.Dinardo (mail 29.05.20)
    theta_x_rad: float = np.radians(1.338)
    # 3dB beamwidth across track [rad], from S.Dinardo (mail 29.05.20)
    theta_y_rad: float = np.radians(1.338)
    # Pulse repetition frequency [Hz] (Determined by reverse engineering,
    # confirmed by W. Smith), from S.Dinardo (mail 02.06.20), 4488 = pulse
    # repetition interval
    prf_Hz: float = 1 / (4488 / 80e6)

    # tau_u: float = 44.8e-6  # useful pulse length [s]
    n_b: int = 64  # number of pulses per Burst [-]
    # Burst Repetition Interval [s] (Determined by reverse engineering,
    # confirmed by W. Smith)
    bri: float = 1018710 * 1 / 80e6

    sh_x: float = 1.0  # antenna shape factor along track [-]
    sh_y: float = 1.0  # antenna shape factor across track [-]

    def get_default_sets(st: SensorType):
        sensor_sets = SensorSettings(sensor_type=st)

        if st is SensorType.S3:
            pass
        elif st is SensorType.CS:
            sensor_sets.sensor_type = SensorType.CS
            sensor_sets.B_r_Hz = 320e6
            sensor_sets.theta_x_rad = np.radians(1.06)
            sensor_sets.theta_y_rad = np.radians(1.1992)
            sensor_sets.prf_Hz = 1 / (4400 * (1 / 80e6))
            sensor_sets.bri = None
        elif "s6" in st.value:
            sensor_sets.sensor_type = st
            sensor_sets.B_r_Hz = 395e6
            theta_x_y = 1.33
            sensor_sets.theta_x_rad = np.radians(theta_x_y)
            sensor_sets.theta_y_rad = np.radians(theta_x_y)
            sensor_sets.prf_Hz = None
            sensor_sets.bri = None

        return sensor_sets


SENSOR_SETS_DEFAULT_S3 = SensorSettings.get_default_sets(SensorType.S3)


class WaveformSettings(BaseModel):
    zp_oversampling_factor: int = 1
    internal_oversampling_factor: int = 1
    # np: Optional[int]
    np: int = None

    def get_default_src_type(st: L1bSourceType, **kwargs):
        if st is L1bSourceType.GPOD:
            return WaveformSettings(
                **{**{"zp_oversampling_factor": 2, "np": 256, **kwargs}}
            )
        elif st is L1bSourceType.EUM_S3:
            return WaveformSettings(
                **{**{"zp_oversampling_factor": 1, "np": 128, **kwargs}}
            )
        elif st is L1bSourceType.EUM_CS:
            return WaveformSettings(
                **{**{"zp_oversampling_factor": 2, "np": 128, **kwargs}}
            )
        elif "s6" in str(st).lower():
            return WaveformSettings(
                **{**{"zp_oversampling_factor": 2, "np": 256, **kwargs}}
            )

    @model_validator(mode="after")
    def set_def_np(self):
        np = self.np if self.np is not None else 128
        self.np = np * self.zp_oversampling_factor * self.internal_oversampling_factor
        return self


class RetrackerProcessorSettings(BaseModel):
    n_offset: int = 0
    n_inds: int = 5  # 0 = all
    n_procs: int = None  # None = all available
    nc_dest_dir: Path = Path().cwd().parent / ".testrun"

    do_interp_dist2coast: bool = False
    do_write_out_nc: bool = True
    do_write_out_log: bool = False
    do_create_settings_log_file: bool = True
    skip_if_exists: bool = True

    do_dynamic_fg_epoch: bool = False
    dynamic_fg_epoch_n_adjacent_meas: int = 20

    auto_detect_s6_rmc: bool = True

    reduce_l2_factor: int = 1


class RetrackerSettings(BaseModel):
    settings_preset: SettingsPreset
    # flag to disable the usage of multilook (0 -> Enable ML; 1 -> Disable
    # Multilook)
    Disable_ML_Flag: bool = False
    # Flag to control activation of waveform normalization (1 = true ; 0 =
    # false);
    Wf_Norm_Flag: bool = True
    # Wf_Norm_Aw: int = 1  # Width (in gates) of the sliding window for waveform normalization
    # Wf_Norm_First: int = 1  # First gate of the waveform to search for the max. amplitude
    # Wf_Norm_Last: int = None  # Last gate of the waveform to search for the
    # max. amplitude
    Wf_TN_First: int = 5  # First gate of the waveform to estimate the amplitude of the thermal noise floor, without zero-padding factor
    Wf_TN_Last: int = 10  # Last gate of the waveform to estimate the amplitude of the thermal noise floor, without zero-padding factor
    # -1 means fit from beginning of waveform, without zero-padding factor
    Wf_Fit_First_Bin: int = -1
    # -1 means fit until end of waveform, without zero-padding factor
    Wf_Fit_Last_Bin: int = -1
    # Wf_Fit_Activate_Flag = 1 #Flag to control the activation of the waveform
    # fitting (1 = true ; 0 = false);
    Thr: float = (
        3.0  # Threshold parameter greater than 1 used to test Wf_max against TN
    )
    # TN_Flag = 0 #flag to determine how to deal with Thermal Noise (TN_Flag=1
    # -> retracked ; TN_Flag=0 constant/estimated)

    second_retracking_step_samplus: bool = False
    # 0:disabled, >0:number of samples around dynamic_fg_epoch
    normalise_wf_by_fg_region: int = 0
    # increases the weights for the detected leading edge
    leading_edge_weight_factor: float = 1.0

    interference_masking: bool = False
    interference_masking_grow: int = 0
    interference_masking_swh_max: float = 18.0
    interference_masking_mask_before_le: bool = False
    interference_masking_second_retracking_step: bool = False

    # auto-detects RAW/RMC mode for S6 and adjusts
    # Wf_Fit_First_Bin/Wf_Fit_Last_Bin
    auto_detect_s6_rmc: bool = True

    subwaveform_mode: bool = False
    subwaveform_n_gates_after_le: int = 5

    # fits the zero-doppler beam (reasonable for FF-SAR-processed waveforms)
    fit_zero_doppler: bool = False

    def get_default_sets(st: SensorType, **kwargs):
        retrack_sets = RetrackerSettings(settings_preset=SettingsPreset.NONE, **kwargs)

        if st is SensorType.S3:
            pass
        elif st is SensorType.CS:
            retrack_sets.Wf_TN_First = 3  # 0-based
            retrack_sets.Wf_TN_Last = 8  # 0-based
        elif "s6" in st.value:
            retrack_sets.Wf_TN_First = 13  # 0-based
            retrack_sets.Wf_TN_Last = 17  # 0-based
            retrack_sets.Wf_Fit_First_Bin = 11  # 0-based
            retrack_sets.Wf_Fit_Last_Bin = 132  # 0-based

            if "ff" in st.value:
                retrack_sets.fit_zero_doppler = True

        return retrack_sets


class ModelParameter(BaseModel):
    lat_rad: float = np.radians(48.0)
    alt_m: float = 815000
    Vs_m_per_s: float = 7500
    h_rate_m_per_s: float = 0.06
    ascending: bool = True
    ksix_rad: float = 0.0
    ksiy_rad: float = 0.0
    beam_ang_stack_rad: np.ndarray = np.array([])
    epoch_ref_gate: float = 65.0
    dist2coast: float = 42.0
    rip: np.ndarray = np.array([])
    pri_hz: Optional[float] = None
    stack_mask_start_stop: Optional[np.ndarray] = None

    @model_validator(mode="before")
    def warning_defaults(cls, data):
        fields_not_set = [f for f in cls.model_fields if f not in data]
        if fields_not_set:
            logging.debug(
                "WARNING: the following params were not set, now taking defaults: {}".format(
                    ",".join(fields_not_set)
                )
            )
        return data

    class Config:
        arbitrary_types_allowed = True


class ModelSettings(BaseModel):
    # flag to disable the computation of first order term in the SAMOSA model
    # (0 -> Enable First Order term; 1 -> Disable First order term)
    Disable_SAM_F1_Flag: bool = False
    # flag to disable the computation of alph_P by LUT (0 -> Use LUT to
    # extract the alph_P value; 1 -> use a constant value for alph_P)
    Disable_LUT_Flag: bool = False
    # Index for the first idealized beam accumulated in the multilooking stage
    Ideal_First_Look_Index_ML: int = -106
    # Index for the last idealized beam accumulated in the multilooking stage
    Ideal_Last_Look_Index_ML: int = 106
    # 1 -> idealized beam angles are computed from simple geometric
    # considerations; 0 -> Use the values from data.
    Ideal_Beam_Ang_Stack_flag: bool = False
    # flag to enable the slope effect compensation (1 -> Enable Slope Effect
    # Compensation; 0 -> Disable Slope Effect Compensation)
    Enable_Slope_Effect_Flag: bool = False
    # as in Dinardo2020, not as in DPM 2.5.2: 1 / (0.886 * np.sqrt(2 * np.pi))
    # # ~0.45, PTR Gaussian approximation coefficient value
    alpha_p_mean: float = 0.5
    # fits the zero-doppler beam (reasonable for FF-SAR-processed waveforms)
    fit_zero_doppler: bool = False

    def get_default_sets(st: SensorType, **kwargs):
        model_sets = ModelSettings(**kwargs)

        if st is SensorType.S3:
            pass
        elif st is SensorType.CS:
            model_sets.alpha_p_mean = 1 / (0.886 * np.sqrt(2 * np.pi))
        elif "s6" in st.value:
            model_sets.alpha_p_mean = 0.55
            model_sets.Enable_Slope_Effect_Flag = False

        return model_sets


class FittingParameters(BaseModel):
    fit_nu_instead_of_swh: bool = False
    swh_first_step: float = None
    init_epoch: tuple = None
    init_swh: tuple = None
    init_pu: tuple = None
    init_nu: tuple = None
    minmax_epoch: tuple = None
    minmax_swh: tuple = None
    minmax_pu: tuple = None
    minmax_nu: tuple = None
    disable_interference_masking: bool = False


class FittingSettings(BaseModel):
    # Initial guess for the 2nd fitted variable (Hs)
    Fit_Var_2_Init_Hs: float = 2.0
    # Initial guess for the 3rd fitted variable (Pu)
    Fit_Var_3_Init_Pu: float = 1.0
    # Initial guess for the 4th fitted variable (ThNoise). Activge if
    # TN_Flag=1.
    Fit_Var_4_Init_TN: float = 0.0
    # Initial guess for the optional fitted variable (nu).
    Fit_Var_5_Init_nu: float = 0
    # Minimum/maximum allowed value for the 1st fitted variable (t0) in
    # nanoseconds
    Fit_Var_1_MinMax_epoch: tuple = (-600.0, 600.0)
    # Minimum/maximum allowed value for the 2nd fitted variable (Hs)
    Fit_Var_2_MinMax_Hs: tuple = (-0.5, 20)
    # Minimum/maximum allowed value for the 3rd fitted variable (Pu)
    Fit_Var_3_MinMax_Pu: tuple = (0.2, 1.5)
    # Minimum/maximum allowed value for the 4th fitted variable (TN).
    Fit_Var_4_MinMax_TN: tuple = (0.0, 0.05)
    # Minimum/maximum allowed value for the optional fitted variable (nu)
    Fit_Var_5_MinMax_nu: tuple = (0, 1e9)
    Fit_MaxIter: int = 20  # Maximum Number of iterations for the fitting routine}
    Levmar_Control_0: float = 1e-3  # Scalar factor of Damping
    Levmar_Control_1: float = 1e-3  # Threshold for Infinite-Norm of Error Gradient
    Levmar_Control_2: float = 1e-2  # Threshold for 2-Norm of distance from solution
    Levmar_Control_3: float = 1e-6  # Threshold for 2-Norm of Error (RMSE)
    Levmar_Control_4: float = 1e-3  # Step used in finite differences for jacobian
    # Threshold for xtol, ftol, and gtol tolerances of TRF fitting algorithm
    trf_threshold: float = 1e-5
    # Threshold for xtol, ftol, and gtol tolerances of TRF fitting algorithm
    trf_stepsize: float = 1e-2
    # sets strict epoch boundaries in gates around fg epoch (0: disabled)
    limit_epoch_around_fg_n: int = 0

    def get_default_sets(st: SensorType, **kwargs):
        fitting_sets = FittingSettings(**kwargs)

        if st is SensorType.S3:
            pass
        elif st is SensorType.CS:
            fitting_sets.trf_stepsize = 1e-2
        elif "s6" in st.value:
            pass

        return fitting_sets


class ExportSettings(BaseModel):
    rp_sets: RetrackerProcessorSettings
    retrack_sets: RetrackerSettings
    fitting_sets: FittingSettings
    sensor_sets: SensorSettings


class ProcMode(Enum):
    FFSAR = "ffsar"
    UFSAR = "ufsar"
