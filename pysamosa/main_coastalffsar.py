import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pysamosa import simple_logger
from pysamosa.common_types import L1bSourceType, ProcMode, SettingsPreset
from pysamosa.data_access import data_vars_dart
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings

simple_logger.set_root_logger(log_level=logging.INFO)


def convert_dt(dt_str):
    return np.datetime64(datetime.strptime(dt_str, "%Y%m%dT%H%M%S"))


TUDTUM_PATH = Path.home() / "TUDTUM"
COASTAL_FFSAR_PATH = Path("/lfs/DGFI24/bigdata/coastal_ffsar/")
# COASTAL_FFSAR_PATH = Path('/nfs/DGFI145/C/work_flo/coastal_ffsar/')

proc_configs = [
    # ffsar
    ((20, 20), "ffsar_dart", ProcMode.FFSAR, 2.1, data_vars_dart["l1b_ffsar"]),
    ((60, 20), "ffsar_dart", ProcMode.FFSAR, 2.1, data_vars_dart["l1b_ffsar"]),
    # ((100,20), 'ffsar_dart', ProcMode.FFSAR, 2.1, data_vars_dart['l1b_ffsar']),
    ((140, 20), "ffsar_dart", ProcMode.FFSAR, 2.1, data_vars_dart["l1b_ffsar"]),
    # ufsar
    ((20, 20), "ufsar_dart", ProcMode.UFSAR, 2.4, data_vars_dart["l1b"]),
    # ((60,20), 'ufsar_dart', ProcMode.UFSAR, 2.4, data_vars_dart['l1b']),
    # ((100,20), 'ufsar_dart', ProcMode.UFSAR, 2.4, data_vars_dart['l1b']),
    # ufsar-eum
    # ((20,20), 'ufsar_eum', ProcMode.UFSAR, 2.4, data_vars_eumetsat_s6['l1b']),
]

skip_if_exists = False
reverse_l1b_file_list = False

if __name__ == "__main__":
    # cycle_pass_pattern = '0(0[5-9]|[1-3][0-9]|4[0-2])_(120)'
    cycle_pass_pattern = "0(0[5-9]|[1-3][0-9]|4[0-2])_(213|018|196|120|044)"
    # cycle_pass_pattern = '0(0[5-9]|[1-3][0-9]|4[0-2])_(213)'
    # cycle_pass_pattern = '0(12|[2-3][0-9]|4[0-2])_(120)'
    # cycle_pass_pattern = '042_(213|018|196|120)'
    # cycle_pass_pattern = '026_120'

    for l1b_l2_posting_rate, procname, mode, T, data_vars in proc_configs:
        l1b_l2_path_append = f'_{procname.split("_")[-1]}' if "opt" in procname else ""
        l1b_posting_rate, l2_posting_rate = l1b_l2_posting_rate
        l1b_src_path_dart = (
            COASTAL_FFSAR_PATH
            / "processed"
            / "l1b"
            / f"T_{str(T).replace('.','-')}_posting_rate_{l1b_posting_rate}"
        )
        # l1b_src_path_eum = Path('/nfs/DGFI145/A/original_data/sentinel6a/JASON_CS_S6A_L1B_ALT_HR_NTC_F/')
        l1b_src_path_eum = Path(
            "/lfs/DGFI24/bigdata/coastal_ffsar/orig_data/f06/P4_1B_HR_____/"
        )

        # l2_destpath = COASTAL_FFSAR_PATH / f'coral_processed_L2_{l1b_posting_rate}_{l2_posting_rate}'
        l2_destpath = (
            COASTAL_FFSAR_PATH
            / "processed"
            / "l2"
            / f"{l1b_posting_rate}_{l2_posting_rate}"
        )

        if "dart" in procname and l1b_l2_path_append:
            data_vars["wf"] = f"power_waveform{l1b_l2_path_append}"

        l1b_files_dart = [
            f
            for f in sorted(l1b_src_path_dart.rglob("*.nc"))
            if "1B" in str(f) and bool(re.search(cycle_pass_pattern, str(f)))
        ]
        l1b_files_eum = [
            f
            for f in sorted(l1b_src_path_eum.rglob("*.nc"))
            if "1B" in str(f) and bool(re.search(cycle_pass_pattern, str(f)))
        ]
        l1b_files = l1b_files_dart if "dart" in procname else l1b_files_eum

        logging.info(
            f"Processing: l1b_l2_posting_rate: {l1b_l2_posting_rate}, len(l1b_files):{len(l1b_files)}, l1b_path:{l1b_src_path_dart} , l2_path:{l2_destpath}"
        )

        cycles = [int(f.name.split("_")[13]) for f in l1b_files_dart]
        passes = [int(f.name.split("_")[14]) for f in l1b_files_dart]
        start_dates = [convert_dt(f.name.split("_")[10]) for f in l1b_files_dart]
        end_dates = [convert_dt(f.name.split("_")[11]) for f in l1b_files_dart]
        bbox = [
            (float(f.stem.split("_")[21]), float(f.stem.split("_")[22]), 0, 360)
            for f in l1b_files_dart
        ]

        df_dart = pd.DataFrame(
            {
                "file": l1b_files_dart,
                "cycle": cycles,
                "ppass": passes,
                "start_date": start_dates,
                "end_date": end_dates,
                "dom": bbox,
            }
        )

        if reverse_l1b_file_list:
            df_dart = df_dart.iloc[::-1]

        is_ffsar = mode == ProcMode.FFSAR

        if is_ffsar:
            l1b_src_type = (
                L1bSourceType.EUM_S6_F04_FFSAR
                if "f04" in str(l1b_files_dart[0]).lower()
                else L1bSourceType.EUM_S6_F06_FFSAR
            )
        else:
            l1b_src_type = (
                L1bSourceType.EUM_S6_F04
                if "f04" in str(l1b_files_dart[0]).lower()
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

        rp_sets.nc_dest_dir = l2_destpath / procname
        rp_sets.n_offset = 0
        rp_sets.n_inds = 0
        rp_sets.n_procs = 20
        rp_sets.skip_if_exists = skip_if_exists
        rp_sets.reduce_l2_factor = round(l1b_posting_rate / l2_posting_rate)
        rp_sets.dynamic_fg_epoch_n_adjacent_meas *= rp_sets.reduce_l2_factor

        additional_nc_attrs = {
            "L1B source type": l1b_src_type.value.upper(),
            "Retracker preset": pres.value.upper(),
        }

        rp = RetrackerProcessor(
            l1b_source=l1b_files,
            l1b_data_vars=data_vars,
            bbox=bbox,
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
