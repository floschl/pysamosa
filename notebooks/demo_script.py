from pathlib import Path

import numpy as np

from pysamosa.common_types import L1bSourceType
from pysamosa.data_access import data_vars_s3, data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import SettingsPreset, get_default_base_settings

from pysamosa.download_aux_data import download_test_data

l1b_files = []
l1b_files.append(
    Path.cwd().parent
    / "pysamosa"
    / ".testdata"
    / "s6"
    / "l1b"
    / "S6A_P4_1B_HR______20211120T051224_20211120T060836_20220430T212619_3372_038_018_009_EUM__REP_NT_F06.nc"
)

if not l1b_files[0].exists():
    test_data_path = download_test_data()

l1b_src_type = L1bSourceType.EUM_S6_F06
data_vars = data_vars_s6

# configure coastal CORAL retracker
pres = SettingsPreset.CORALv2
rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(
    settings_preset=pres, l1b_src_type=l1b_src_type
)

rp_sets.nc_dest_dir = l1b_files[0].parent / "processed"
rp_sets.n_offset = 0
rp_sets.n_inds = 0  # 0 means all
rp_sets.n_procs = 6  # use 6 cores in parallel
rp_sets.skip_if_exists = False

additional_nc_attrs = {
    "L1B source type": l1b_src_type.value.upper(),
    "Retracker preset": pres.value.upper(),
}

rp = RetrackerProcessor(
    l1b_source=l1b_files,
    l1b_data_vars=data_vars["l1b"],
    rp_sets=rp_sets,
    retrack_sets=retrack_sets,
    fitting_sets=fitting_sets,
    wf_sets=wf_sets,
    sensor_sets=sensor_sets,
    nc_attrs_kw=additional_nc_attrs,
    bbox=[np.array([-29.05, -29.00, 0, 360])],
)

rp.process()  # start processing

print(rp.output_l2)  # retracked L2 output can be found in here
