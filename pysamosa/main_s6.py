import logging
from pathlib import Path

from pysamosa.common_types import L1bSourceType, SettingsPreset
from pysamosa.data_access import data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings

if __name__ == "__main__":
    nc_src_base_path = Path(
        "/nfs/DGFI145/A/original_data/sentinel6a/JASON_CS_S6A_L1B_ALT_HR_NTC_F_V2/"
    )
    # nc_dest_path = Path('/nfs/DGFI145/C/work_flo/s6jtex_coral/')
    run_name = "coral_retrack"
    nc_dest_path = nc_src_base_path.parent

    # select files
    l1b_files = [f for f in sorted(nc_src_base_path.rglob("*.nc"))]

    l1b_src_type = (
        L1bSourceType.EUM_S6_F04
        if "f04" in str(l1b_files[0]).lower()
        else L1bSourceType.EUM_S6_F06
    )
    pres = SettingsPreset.CORALv2
    # pres = SettingsPreset.NONE
    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(settings_preset=pres, l1b_src_type=l1b_src_type)

    rp_sets.nc_dest_dir = nc_dest_path / run_name
    rp_sets.n_offset = 0
    rp_sets.n_inds = 0
    rp_sets.n_procs = 6
    rp_sets.skip_if_exists = False

    additional_nc_attrs = {
        "L1B source type": l1b_src_type.value.upper(),
        "Retracker preset": pres.value.upper(),
    }

    rp = RetrackerProcessor(
        l1b_source=l1b_files,
        l1b_data_vars=data_vars_s6["l1b"],
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        wf_sets=wf_sets,
        sensor_sets=sensor_sets,
        nc_attrs_kw=additional_nc_attrs,
        log_level=logging.DEBUG,  # comment in to show debug messages
    )
    rp.process()
