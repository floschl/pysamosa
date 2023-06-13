import pytest
from pathlib import Path
import re
import netCDF4
from enum import Enum
import matplotlib as mpl

from pysamosa.download_aux_data import download_test_data


mpl.use("Agg")
# mpl.use('TkAgg')

collect_ignore = ["setup.py"]
collect_ignore_glob = ["*_montecarlo_sim.py"]

from pysamosa.data_access import (
    _read_dataset_vars_from_ds,
    data_vars_s3,
    data_vars_s6,
    data_vars_cs,
    data_vars_dart,
)

from pysamosa.settings import S3_DATA_DIR, S6_DATA_DIR, CS_DATA_DIR, FFSAR_DATA_DIR


class FileLevel(Enum):
    L1A = "1A"
    L1B = "1B"
    L2 = "2"


# file-id to regex mappings
file_id_mappings = {
    "s3_0": "S3A_(.*)_(.{4})_030_090",
    "s6_eum_0_f03": "S6A_P4_1B_(.*)_030_018_(.*)_F03",  # crete track
    "s6_eum_1_f04": "S6A_P4_1B_(.*)_038_018_(.*)_F04",  # crete track
    "s6_eum_2_f06": "S6A_P4_1B_(.*)_038_018_(.*)_F06",  # crete track
    "cs_0": "CS_LTA__SIR_SAR_1B_20150820T165618_20150820T165810_D001",
    "cs_1": "CS_LTA__SIR_SAR_1B_20150503T160800_20150503T161329_D001",
    "cs_2": "CS_LTA__SIR_GOPR1B_20150531T153351_20150531T153920_C001",
    "dart_s6_0": "S6A_P4_1B_HR______20211101T093654_20211101T103310_20211127T045347_3376_036_044_022_DAR__OPE_NT_F04",
}


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    download_test_data()


class DatasetReader:
    def __init__(self, src, data_vars, group="", file_level=None):
        self.data_vars = data_vars
        self.group = group
        self.file_level = file_level

        if src.is_dir():
            self.ncfiles = sorted(
                list(src.rglob("*.nc")),
                key=lambda f: str(f.name).replace("RES_", "")[16:31],
            )
        else:
            self.ncfiles = Path(src)

    def __call__(self, *, n_offset=0, n_inds=1, file_id=None):
        ncfile = self.get_nc_filename(file_id)

        return _read_dataset_vars_from_ds(
            nc_filename=ncfile,
            data_var_names=self.data_vars,
            n_offset=n_offset,
            n_inds=n_inds,
            group=self.group,
        )

    def get_nc_filename(self, file_id=None):
        try:
            if file_id and isinstance(self.ncfiles, list):
                file_pattern = (
                    file_id_mappings[file_id]
                    if not self.file_level
                    else file_id_mappings[file_id].replace(
                        "1B", self.file_level.value.ljust(2, "_")
                    )
                )
                nc_filename = [
                    nc
                    for nc in self.ncfiles
                    if bool(re.match(f"(.*){file_pattern}(.*)", str(nc)))
                ][0]
            else:
                nc_filename = self.ncfiles
        except IndexError:
            raise IndexError(f"{file_id_mappings[file_id]} cannot be found. ")

        return nc_filename

    # def __getitem__(self, item):
    #     return self.ncfile_list[item]
    #     return self.ncfile_list[item]


@pytest.fixture
def s3_eum_l1b():
    return DatasetReader(src=S3_DATA_DIR / "l1b", data_vars=data_vars_s3["l1b"])


@pytest.fixture
def s3_eum_l2():
    return DatasetReader(src=S3_DATA_DIR / "l2", data_vars=data_vars_s3["l2"])


@pytest.fixture
def s6_eum_l1b():
    return DatasetReader(
        src=S6_DATA_DIR / "l1b", data_vars=data_vars_s6["l1b"], group="data_20/ku"
    )


@pytest.fixture
def s6_eum_l2():
    return DatasetReader(
        src=S6_DATA_DIR / "l2",
        data_vars=data_vars_s6["l2"],
        group="data_20/ku",
        file_level=FileLevel.L2,
    )


@pytest.fixture
def s6_dart_l1b():
    return DatasetReader(
        src=S6_DATA_DIR / "l1b", data_vars=data_vars_dart["l1b"], group=""
    )


@pytest.fixture
def s6_dart_l2():
    return DatasetReader(
        src=FFSAR_DATA_DIR / "l2",
        data_vars=data_vars_dart["l2"],
        group="",
        file_level=FileLevel.L2,
    )


@pytest.fixture
def cs_eum_l1b():
    return DatasetReader(src=CS_DATA_DIR / "l1b", data_vars=data_vars_cs["l1b"])


@pytest.fixture
def cs_eum_l2():
    return DatasetReader(
        src=CS_DATA_DIR / "l2", data_vars=data_vars_cs["l2"], file_level=FileLevel.L2
    )


@pytest.fixture
def dataset_generic_l1b():
    def _dataset_reader(ncfile, data_vars):
        grps = [
            f"{g}/ku" for g in list(netCDF4.Dataset(ncfile).groups) if "data_20" in g
        ]
        return DatasetReader(
            src=ncfile,
            data_vars=data_vars,
            file_level=FileLevel.L1B,
            group=grps[0] if len(grps) else "",
        )

    return _dataset_reader


@pytest.fixture
def dataset_generic_l2():
    def _dataset_reader(ncfile, data_vars):
        grps = [
            f"{g}/ku" for g in list(netCDF4.Dataset(ncfile).groups) if "data_20" in g
        ]
        return DatasetReader(
            src=ncfile,
            data_vars=data_vars,
            file_level=FileLevel.L2,
            group=grps[0] if len(grps) else "",
        )

    return _dataset_reader
