import logging
import os
import re
from pathlib import Path
from sys import platform

from pysamosa.common_types import L1bSourceType, SettingsPreset
from pysamosa.data_access import data_vars_s3
from pysamosa.retracker_processor import RetrackerProcessor
from pysamosa.settings_manager import get_default_base_settings

is_linux = "linux" in platform
is_slurm = "SLURM_JOB_ID" in os.environ
slurm_abs_procid, slurm_n_total_processes = None, None
if is_slurm:
    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        slurm_n_array_jobs = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        slurm_array_job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

        slurm_n_processes_per_job = (
            int(os.environ["SLURM_NPROCS"]) if "SLURM_NPROCS" in os.environ else 1
        )
        n_process_id = int(os.environ["SLURM_PROCID"])

        slurm_n_total_processes = slurm_n_array_jobs * slurm_n_processes_per_job
        slurm_abs_procid = slurm_array_job_id * slurm_n_processes_per_job + n_process_id
    else:
        slurm_n_total_processes = int(os.environ["SLURM_NPROCS"])
        slurm_abs_procid = int(os.environ["SLURM_PROCID"])

    print(
        f"slurm_abs_procid/slurm_n_total_processes: ({slurm_abs_procid}/{slurm_n_total_processes})"
    )

process_rr_files = True

top_coastal_tracks = [
    "371",
    "478",
    "592",
    "074",
    "463",
    "207",
    "508",
    "250",
    "514",
    "491",
    "741",
    "355",
    "181",
    "285",
    "319",
    "493",
    "530",
    "115",
    "703",
    "226",
    "656",
    "513",
    "084",
    "099",
    "646",
    "142",
    "011",
    "393",
    "695",
    "474",
]
rr_track_nums = f'({"|".join(top_coastal_tracks)})'
# rr_track_nums = f'({"|".join(top_coastal_tracks[:1])})'
# rr_track_nums = f'(?!{"|".join(top_coastal_tracks[:5])})[0-9][0-9][0-9]'

name_dest_path = "coral_paper"

# dest_path_base = Path.home() / 'dss' if is_slurm else Path('/nfs/public_ads/Schlembach')
dest_path_base = Path.home() / "dss" if is_slurm else Path("/nfs/DGFI8/H/work_flo/")
dest_path_base = dest_path_base / "pysamosa_results" / name_dest_path

l1bsrc_type, preset = (
    L1bSourceType.GPOD,
    SettingsPreset.CORALv1,
)

# round robin files config
if preset == SettingsPreset.NONE:
    l1b_base_path = (
        "s3a_sr_1_sra_bs"
        if l1bsrc_type is L1bSourceType.EUM_S3
        else "s3a_sr_1_sra_bs_gpod_sam_wfs"
    )
elif preset == SettingsPreset.SAMPLUS:
    l1b_base_path = "s3a_sr_1_sra_bs_gpod_samplus_wfs"
# rr_l1b_src_dir = (Path('/nfs/') if is_linux else Path('U:/')) / 'DGFI145/C/seastate_cci/round-robin/satellite/' / l1b_base_path
rr_l1b_src_dir = (
    (Path("/nfs/") if is_linux else Path("U:/"))
    / "DGFI145/C/seastate_cci/round-robin/satellite/"
    / l1b_base_path
)
# rr_l1b_src_dir = (Path('/nfs/') if is_linux else Path('U:/')) / 'DGFI8/H/work_flo/lfsdata/' / l1b_base_path
# rr_l1b_src_dir = Path('/tmp/lfsdata/') / l1b_base_path
# rr_l1b_src_dir = Path('/lfs/DGFI24/data/schlembach/pysamosa_lfsdata/') / l1b_base_path
if is_slurm:
    rr_l1b_src_dir = Path.home() / "dss" / "seastate_cci_rr_lfsdata/" / l1b_base_path

n_offset = 0
n_inds = 0

# n_procs = None
n_procs = 8

# if commented out, RR files will be processed
# src_path = Path('/nfs/DGFI36/altimetry/seastate_cci/round-robin/satellite/s3a_sr_1_sra_bs/')
# l1b_files = [
#     src_path / 'track_207/S3A_SR_1_SRA_BS_20170911T025353_20170911T034422_20171006T170509_3029_022_103______MAR_O_NT_002.SEN3/measurement_l1bs.nc',
#     src_path / 'track_463/S3A_SR_1_SRA_BS_20170411T022013_20170411T031042_20170506T171306_3029_016_231______MAR_O_NT_002.SEN3/measurement_l1bs.nc',
# ]
# l1b_files = [
#     Path('/lfs/DGFI24/fastdata/schlembach/repos/pysamosa/lfsdata/eumetsat/l1bs/S3A_SR_1_SRA_BS_20171210T145735_20171210T154803_20190705T205005_3028_025_239______MR1_R_NT_004.nc'),
# ]


def get_l1b_src_files(*, nc_dest_path, skip_if_exists=True):
    l1b_files = [
        f
        for f in sorted(Path(rr_l1b_src_dir).rglob("*.nc"))
        if bool(re.search(f"(track_){rr_track_nums}", str(f)))
    ]

    if is_slurm:
        # remove files from list if they already exist
        if skip_if_exists:
            l1b_files = [
                f
                for f in l1b_files
                if not (nc_dest_path / str(f.relative_to(rr_l1b_src_dir))).exists()
            ]

        def split(a, n):
            k, m = divmod(len(a), n)
            return (
                a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        # split file list into slurm_n_procs chunks of files, take
        # slurm_procid's chunk
        l1b_files = list(split(l1b_files, slurm_n_total_processes))[slurm_abs_procid]

    return l1b_files[::-1]


if __name__ == "__main__":
    if (preset == SettingsPreset.SAMPLUS) and l1bsrc_type == L1bSourceType.EUM_S3:
        raise RuntimeError(
            "SettingsPreset SAM+/SAM++ and L1bSourceType.EUMETSAT is not compatible. "
        )

    nc_dest_path = dest_path_base / (l1bsrc_type.value + "_" + preset.value)

    (
        rp_sets,
        retrack_sets,
        fitting_sets,
        wf_sets,
        sensor_sets,
    ) = get_default_base_settings(settings_preset=preset, l1b_src_type=l1bsrc_type)

    rp_sets.nc_dest_dir = nc_dest_path
    rp_sets.n_offset = n_offset
    rp_sets.n_inds = n_inds
    rp_sets.n_procs = n_procs
    # rp_sets.skip_if_exists = False

    l1b_data_vars = data_vars_s3["l1b"]
    if "l1b_files" not in locals():
        l1b_files = get_l1b_src_files(
            nc_dest_path=rp_sets.nc_dest_dir, skip_if_exists=rp_sets.skip_if_exists
        )

    additional_nc_attrs = {
        "L1B source type": l1bsrc_type.value.upper(),
        "Retracker preset": preset.value.upper(),
    }

    rp = RetrackerProcessor(
        l1b_source=l1b_files,
        l1b_data_vars=l1b_data_vars,
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
