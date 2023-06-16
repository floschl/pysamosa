import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pysamosa.common_types import L1bSourceType, SettingsPreset
from pysamosa.data_access import data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings


def convert_dt(dt_str):
    return np.datetime64(datetime.strptime(dt_str, "%Y%m%dT%H%M%S"))


if __name__ == "__main__":
    nc_src_base_path = Path(
        "/nfs/DGFI145/A/original_data/sentinel6a/JASON_CS_S6A_L1B_ALT_HR_NTC_F_V2/"
    )
    run_name = "s6jtex_coral_reprocessed"
    nc_dest_path = Path("/nfs/DGFI145/C/work_flo/s6jtex_coral/")

    # cycle_pass_pattern = '_013_\d{3}_\d{3}_'
    cycle_pass_pattern = "_0(1[3-9]|2[0-2])_\\d{3}_\\d{3}_"
    # cycle_pass_pattern = '_022_\d{3}_\d{3}_'
    # cycle_pass_pattern = '_017_013_\d{3}_'

    # select files
    l1b_files = [
        f
        for f in sorted(nc_src_base_path.rglob("*.nc"))
        if bool(re.search(cycle_pass_pattern, str(f)))
    ]

    l1b_src_type = (
        L1bSourceType.EUM_S6_F04
        if "f04" in str(l1b_files[0]).lower()
        else L1bSourceType.EUM_S6_F06
    )
    pres = SettingsPreset.CORALv2
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
    rp_sets.n_procs = 40
    rp_sets.skip_if_exists = True

    cycles = [int(f.name.split("_")[13]) for f in l1b_files]
    passes = [int(f.name.split("_")[14]) for f in l1b_files]
    start_dates = [convert_dt(f.name.split("_")[10]) for f in l1b_files]
    end_dates = [convert_dt(f.name.split("_")[11]) for f in l1b_files]

    df_l1b_files = pd.DataFrame(
        {
            "file": l1b_files,
            "cycle": cycles,
            "ppass": passes,
            "start_date": start_dates,
            "end_date": end_dates,
        }
    )
    df_l1b_files = df_l1b_files.drop_duplicates(["cycle", "ppass"])

    additional_nc_attrs = {
        "L1B source type": l1b_src_type.value.upper(),
        "Retracker preset": pres.value.upper(),
    }

    rp = RetrackerProcessor(
        l1b_source=df_l1b_files.file.values,
        l1b_data_vars=data_vars_s6["l1b"],
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
