import logging
import re
from pathlib import Path

import pandas as pd

from pysamosa.common_types import L1bSourceType, SettingsPreset
from pysamosa.data_access import data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings

if __name__ == "__main__":
    nc_dest_path = Path("/lfs/DGFI24/bigdata/s6jtex_raw2rmc/retracked/")

    run_name_basepath = [
        ("coral_raw", Path("/lfs/DGFI24/bigdata/s6jtex_raw2rmc/HR_RAW/1B/")),
        ("coral_rmc", Path("/lfs/DGFI24/bigdata/s6jtex_raw2rmc/HR_RMC/1B/")),
    ]

    l1b_src_type, rbt, pres = (
        L1bSourceType.EUM_S6_F04,
        SettingsPreset.CORALv2,
    )
    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(settings_preset=pres, l1b_src_type=l1b_src_type)

    for run_name, nc_src_base_path in run_name_basepath:
        rp_sets.nc_dest_dir = nc_dest_path / run_name
        rp_sets.n_offset = 0
        rp_sets.n_inds = 0
        rp_sets.n_procs = 4
        rp_sets.skip_if_exists = False

        # select files
        l1b_files = [
            f
            for f in sorted(nc_src_base_path.rglob("*.nc"))
            if bool(
                re.search(
                    "02(5|6|7)_(018|044|120|196|213)_\\d{3}_EUM", str(f.parent.name)
                )
            )
        ]
        # l1b_files = [f for f in sorted(nc_src_base_path.rglob('*.nc')) if bool(re.search('02(5|6|7)_(018|044)_\d{3}_EUM', str(f.parent.name)))]
        df_l1b_files = pd.DataFrame(
            {
                "file": l1b_files,
                "cycle": [int(f.parent.name.split("_")[13]) for f in l1b_files],
                "ppass": [int(f.parent.name.split("_")[14]) for f in l1b_files],
            }
        )
        bboxes_per_ppass = {
            18: (53.65, 53.88, 0, 360),
            44: (50.99, 51.406, 0, 360),
            120: (51.794, 52.05, 0, 360),
            196: (53.13, 53.375, 0, 360),
            213: (53.73, 53.95, 0, 360),
        }
        bboxes_all = 0

        additional_nc_attrs = {
            "L1B source type": l1b_src_type.value.upper(),
            "Retracker preset": pres.value.upper(),
        }

        for p in df_l1b_files.ppass.unique():
            sel_files = df_l1b_files[df_l1b_files.ppass == p]

            rp = RetrackerProcessor(
                l1b_source=sel_files.file.to_list(),
                l1b_data_vars=data_vars_s6["l1b"],
                bbox=[bboxes_per_ppass[p] for i in range(len(sel_files))],
                rp_sets=rp_sets,
                retrack_sets=retrack_sets,
                fitting_sets=fitting_sets,
                wf_sets=wf_sets,
                sensor_sets=sensor_sets,
                nc_attrs_kw=additional_nc_attrs,
                # log_level=logging.DEBUG,  #comment in to
                # show debug messages
            )
            rp.process()

    logging.shutdown()
