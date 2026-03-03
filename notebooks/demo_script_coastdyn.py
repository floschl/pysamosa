#!/usr/bin/env python3
import re
from pathlib import Path
import shutil
import numpy as np

from pysamosa.common_types import L1bSourceType
from pysamosa.data_access import data_vars_s6
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import SettingsPreset, get_default_base_settings


# ======================================================================
# CONFIG
# ======================================================================
input_list = Path("/nfs/ne62rut/TUM-PC/Dokumente/Projects/ESA/Land_Ocean_Interactions_Expro/Directories_S6.txt")
l1b_root = Path("/nfs/DGFI45/F/original_data/sentinel6a/JASON_CS_S6A_L1B_ALT_HR_NTC_F08")
output_dir = Path("/nfs/DGFI144/F/work_marcello/sentinel6a/COASTDYN_SAR")
output_dir.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Helpers
# ======================================================================

def extract_cycle_pass_l2(filename: str):
    """Extract cycle and pass from an L2 filename (fields 9 and 10)."""
    parts = filename.split("_")
    if len(parts) < 11:
        return None, None
    cycle = re.sub(r"\D", "", parts[8])
    pas = re.sub(r"\D", "", parts[9])
    return cycle, pas


def extract_cycle_pass_l1(filename: str):
    """Extract cycle and pass from an L1B filename (fields 14 and 15)."""
    parts = filename.split("_")
    if len(parts) < 16:
        return None, None
    cycle = re.sub(r"\D", "", parts[13])
    pas = re.sub(r"\D", "", parts[14])
    return cycle, pas


# ======================================================================
# Main loop
# ======================================================================
with open(input_list, "r") as f:
    all_l2_paths = [line.strip() for line in f if line.strip()]

for l2_path in all_l2_paths:

    l2_name = Path(l2_path).name
    cycle, pas = extract_cycle_pass_l2(l2_name)

    if not cycle or not pas:
        print(f"[WARN] Cannot parse cycle/pass for L2: {l2_name}")
        continue

    cycle_dir = l1b_root / cycle
    if not cycle_dir.is_dir():
        print(f"[WARN] Missing L1B cycle dir: {cycle_dir}")
        continue

    # ---------------------------------------------------
    # Find matching L1B file
    # ---------------------------------------------------
    matched_l1 = None
    for f in cycle_dir.glob("*.nc"):
        f_cycle, f_pass = extract_cycle_pass_l1(f.name)
        if f_cycle == cycle and f_pass == pas:
            matched_l1 = f
            break

    if matched_l1 is None:
        print(f"[WARN] No L1B match for L2={l2_name} (cycle={cycle}, pass={pas})")
        continue

    # ---------------------------------------------------
    # REPROCESS USING pysamosa
    # ---------------------------------------------------
    print(f"[INFO] Processing L1B: {matched_l1}")

    # Select L1B source type
    try:
        l1b_src_type = L1bSourceType.EUM_S6_G01
    except AttributeError:
        l1b_src_type = L1bSourceType.EUM_S6_F06

    data_vars = data_vars_s6

    pres = SettingsPreset.CORALv2
    rp_sets, retrack_sets, fitting_sets, wf_sets, sensor_sets = get_default_base_settings(
        settings_preset=pres, l1b_src_type=l1b_src_type
    )

    rp_sets.nc_dest_dir = output_dir
    rp_sets.n_offset = 0
    rp_sets.n_inds = 0
    rp_sets.n_procs = 6
    rp_sets.skip_if_exists = False

    additional_nc_attrs = {
        "L1B source type": getattr(l1b_src_type, "value", str(l1b_src_type)).upper(),
        "Retracker preset": pres.value.upper(),
    }

    rp = RetrackerProcessor(
        l1b_source=[matched_l1],
        l1b_data_vars=data_vars["l1b"],
        rp_sets=rp_sets,
        retrack_sets=retrack_sets,
        fitting_sets=fitting_sets,
        wf_sets=wf_sets,
        sensor_sets=sensor_sets,
        nc_attrs_kw=additional_nc_attrs,
        bbox=[np.array([53, 55, 2, 10])],
    )

    rp.process()

    # ---------------------------------------------------
    # Rename output to EXACTLY the same filename as L1B
    # ---------------------------------------------------
    final_name = matched_l1.name
    final_output_path = output_dir / final_name

    produced_files = list(output_dir.glob("*.nc"))
    produced_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = produced_files[0]

    if final_output_path.exists():
        final_output_path.unlink()

    shutil.move(str(latest), str(final_output_path))

    print(f"[OK] Saved as {final_output_path}")


print("=== ALL DONE ===")
