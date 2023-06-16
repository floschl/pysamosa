import logging
from pathlib import Path

from pysamosa.common_types import L1bSourceType, SettingsPreset
from pysamosa.data_access import data_vars_cs
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings

if __name__ == "__main__":
    nc_src_base_path = Path(
        "/nfs/DGFI145/C/work_flo/cs2_files_samplus_test/CS2_open_ocean/"
    )
    nc_dest_path = Path("/nfs/DGFI145/C/work_flo/cs2_files_samplus_test/")
    run_name = "samplus_cs_test"

    l1b_src_type = L1bSourceType.EUM_CS
    pres = SettingsPreset.CORALv1
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
    rp_sets.n_procs = 1
    # rp_sets.skip_if_exists = False

    # select files
    # pat = 'SIR_SAR_1B_20150503T160800'
    # l1b_files = [f for f in nc_src_base_path.rglob('*.nc') if bool(re.match(f'(.*){pat}(.*)', str(f)))]
    l1b_files = (
        open("/nfs/DGFI145/C/work_flo/cs2_files_samplus_test/files_CS2_L1B_D.txt")
        .read()
        .splitlines()
    )
    l1b_files = [Path(f) for f in l1b_files]

    additional_nc_attrs = {
        "L1B source type": l1b_src_type.value.upper(),
        "Retracker preset": pres.value.upper(),
    }

    rp = RetrackerProcessor(
        l1b_source=l1b_files,
        l1b_data_vars=data_vars_cs["l1b"],
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        wf_sets=wf_sets,
        sensor_sets=sensor_sets,
        nc_attrs_kw=additional_nc_attrs,
        # log_level=logging.DEBUG,  #comment in to show
        # debug messages
    )
    rp.process()

    logging.shutdown()
