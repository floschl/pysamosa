import logging

from pysamosa.common_types import L1bSourceType, SettingsPreset
from pysamosa.data_access import data_vars_s3
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings import TEST_DATA_DIR
from pysamosa.settings_manager import get_default_base_settings

if __name__ == "__main__":
    nc_src_base_path = TEST_DATA_DIR / "s3" / "l1b"
    run_name = "s3_retrack_open_ocean"
    nc_dest_path = nc_src_base_path.parent

    # select files
    l1b_files = [f for f in sorted(nc_src_base_path.rglob("*.nc"))]

    l1b_src_type = L1bSourceType.EUM_S3
    # pres = SettingsPreset.CORALv1
    pres = SettingsPreset.NONE
    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(settings_preset=pres, l1b_src_type=l1b_src_type)

    rp_sets.nc_dest_dir = nc_dest_path / run_name
    rp_sets.n_offset = 25800
    rp_sets.n_inds = 100
    rp_sets.n_procs = 6
    rp_sets.skip_if_exists = False

    additional_nc_attrs = {
        "L1B source type": l1b_src_type.value.upper(),
        "Retracker preset": pres.value.upper(),
    }

    rp = RetrackerProcessor(
        l1b_source=l1b_files,
        l1b_data_vars=data_vars_s3["l1b"],
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        wf_sets=wf_sets,
        sensor_sets=sensor_sets,
        nc_attrs_kw=additional_nc_attrs,
        log_level=logging.DEBUG,  # comment in to show debug messages
    )
    rp.process()
