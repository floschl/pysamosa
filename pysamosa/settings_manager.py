from pysamosa.common_types import (
    FittingSettings,
    L1bSourceType,
    RetrackerProcessorSettings,
    RetrackerSettings,
    SensorSettings,
    SensorType,
    SettingsPreset,
    WaveformSettings,
)


def get_default_base_settings(
    *,
    settings_preset: SettingsPreset = SettingsPreset.NONE,
    l1b_src_type: L1bSourceType
):
    if "s3" in l1b_src_type.value:
        sensor_type = SensorType.S3
    elif "cs" in l1b_src_type.value:
        sensor_type = SensorType.CS
    elif "s6" in l1b_src_type.value and "ff" in l1b_src_type.value:
        sensor_type = (
            SensorType.S6_F06_FF
            if "f06" in l1b_src_type.value
            else SensorType.S6_F04_FF
        )
    elif "s6" in l1b_src_type.value and "ff" not in l1b_src_type.value:
        sensor_type = (
            SensorType.S6_F06 if "f06" in l1b_src_type.value else SensorType.S6_F04
        )
    else:
        raise RuntimeError("error setting SensorType. ")

    sensor_sets = SensorSettings.get_default_sets(sensor_type)

    rp_sets = RetrackerProcessorSettings()
    retrack_sets = RetrackerSettings.get_default_sets(st=sensor_type)
    fitting_sets = FittingSettings.get_default_sets(st=sensor_type)

    wf_sets = WaveformSettings.get_default_src_type(l1b_src_type)

    # Settings PRESETS
    if (
        settings_preset == SettingsPreset.CORALv1
        or settings_preset == SettingsPreset.CORALv2
    ):
        rp_sets.do_dynamic_fg_epoch = True
        rp_sets.dynamic_fg_epoch_n_adjacent_meas = 40

        retrack_sets.settings_preset = SettingsPreset.CORALv1
        retrack_sets.second_retracking_step_samplus = True
        retrack_sets.normalise_wf_by_fg_region = 5
        # retrack_sets.leading_edge_weight_factor = 4.0
        retrack_sets.interference_masking = True
        retrack_sets.interference_masking_grow = 5
        retrack_sets.interference_masking_swh_max = 8.0
        retrack_sets.interference_masking_mask_before_le = False
        retrack_sets.interference_masking_second_retracking_step = True

        # fitting_sets.Levmar_Control_2 = 1e-2
        fitting_sets.limit_epoch_around_fg_n = 10
        fitting_sets.trf_threshold = 1e-4
        fitting_sets.trf_stepsize = 1e-2

        if settings_preset == SettingsPreset.CORALv2:
            retrack_sets.settings_preset = SettingsPreset.CORALv2
            retrack_sets.interference_masking_mask_before_le = True
            fitting_sets.Fit_Var_2_MinMax_Hs = (0.0, 20)

    elif settings_preset == SettingsPreset.SAMPLUS:
        retrack_sets.settings_preset = SettingsPreset.SAMPLUS

        rp_sets.do_dynamic_fg_epoch = True
        rp_sets.dynamic_fg_epoch_n_adjacent_meas = 20

        retrack_sets.second_retracking_step_samplus = True

        fitting_sets.trf_threshold = 1e-5
        fitting_sets.trf_stepsize = 1e-2
    elif settings_preset == SettingsPreset.SAMPLUSPLUS:
        retrack_sets.second_retracking_step_samplus = False

    # adapt parameters according to internal_oversampling_factor
    if wf_sets.internal_oversampling_factor > 1 or wf_sets.zp_oversampling_factor > 1:
        mult_factor = (
            wf_sets.internal_oversampling_factor
            if wf_sets.zp_oversampling_factor == 1
            else wf_sets.zp_oversampling_factor
        )

        retrack_sets.normalise_wf_by_fg_region *= mult_factor
        retrack_sets.interference_masking_grow *= mult_factor
        fitting_sets.limit_epoch_around_fg_n *= mult_factor

    return rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets
