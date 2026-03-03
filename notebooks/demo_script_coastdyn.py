from pathlib import Path
import numpy as np
import shutil

from pysamosa.common_types import L1bSourceType
from pysamosa.data_access import data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import SettingsPreset, get_default_base_settings

# -----------------------------
# Configure input (L1B) sources
# -----------------------------
# Base path to your source data (adjust if you want it configurable)
src_data_path = Path("/ADB/original")  # <-- set this root as needed

# Build the L1B paths exactly as you specified
l1b_src_path = src_data_path / "sentinel6a" / "JASON_CS_S6A_L1B_ALT_HR_NTC_G01" / "127"
l1b_file_path = l1b_src_path / "S6A_P4_1B_HR______20240420T170115_20240420T175728_20250611T232717_3373_127_018_009_EUM__REP_NT_G01.SEN6.measurement.nc"

# Ensure the input exists
if not l1b_file_path.exists():
    raise FileNotFoundError(
        f"L1B file not found:\n  {l1b_file_path}\n"
        f"Please verify 'src_data_path' or the filename."
    )

l1b_files = [l1b_file_path]

# --------------------------------
# Select the appropriate source type
# --------------------------------
# Prefer the G01 variant if available in your pysamosa version
try:
    l1b_src_type = L1bSourceType.EUM_S6_G01
except AttributeError:
    # Fallback if your version doesn't expose G01
    l1b_src_type = L1bSourceType.EUM_S6_F06

data_vars = data_vars_s6  # still Sentinel-6

# --------------------------
# Configure retracker preset
# --------------------------
pres = SettingsPreset.CORALv2
rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(
    settings_preset=pres, l1b_src_type=l1b_src_type
)

# --------------------------
# Output directory (your NFS)
# --------------------------
target_out_dir = Path("/nfs/DGFI144/F/work_marcello/sentinel6a/COASTDYN_SAR")
target_out_dir.mkdir(parents=True, exist_ok=True)
rp_sets.nc_dest_dir = target_out_dir

# Processing controls
rp_sets.n_offset = 0
rp_sets.n_inds = 0       # 0 = process all records
rp_sets.n_procs = 6      # parallel workers
rp_sets.skip_if_exists = False  # set True to skip existing outputs

# Extra NetCDF attributes
additional_nc_attrs = {
    "L1B source type": getattr(l1b_src_type, "value", str(l1b_src_type)).upper(),
    "Retracker preset": pres.value.upper(),
}

# --------------------------
# Build and run the processor
# --------------------------
rp = RetrackerProcessor(
    l1b_source=l1b_files,
    l1b_data_vars=data_vars["l1b"],
    rp_sets=rp_sets,
    retrack_sets=retrack_sets,
    fitting_sets=fitting_sets,
    wf_sets=wf_sets,
    sensor_sets=sensor_sets,
    nc_attrs_kw=additional_nc_attrs,
    bbox=[np.array([53, 55, 2, 10])],  # keep or adjust AOI as needed
)

rp.process()



# Desired final output name = exactly the same as the input filename
final_name = l1b_file_path.name
final_output_path = rp_sets.nc_dest_dir / final_name

# Find produced output file
produced_files = list(Path(rp_sets.nc_dest_dir).glob("*.nc"))
if not produced_files:
    raise RuntimeError("No .nc outputs produced.")

# If multiple exist, pick the most recent (typical case: only one file)
produced_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
produced_file = produced_files[0]

# If target already exists, overwrite it
if final_output_path.exists():
    final_output_path.unlink()

# Rename/move
shutil.move(str(produced_file), str(final_output_path))

print(f"Final output saved as: {final_output_path}")
