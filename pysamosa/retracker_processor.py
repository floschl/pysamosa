import logging
import os
from concurrent import futures
from itertools import repeat

import netCDF4
import numpy as np
import xarray as xr

from pysamosa import simple_logger
from pysamosa.common_types import (
    ExportSettings,
    FittingSettings,
    L1bSourceType,
    RetrackerProcessorSettings,
    RetrackerSettings,
    SensorSettings,
    SettingsPreset,
    WaveformSettings,
)
from pysamosa.conf_params import CONST_C
from pysamosa.data_access import (
    _read_dataset_vars_from_ds,
    gen_model_param_obj_from_l1b_data,
    gen_subset_dataset,
    get_model_param_obj_from_l1b_data,
    get_nc_src_dest_file_list,
    get_subset_dataset,
)
from pysamosa.retracker import SamosaRetracker
from pysamosa.retracker_helpers import get_dynamic_first_guess_epochs
from pysamosa.version import __version__


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    dist_km = 6378 * c
    return dist_km


class RetrackerProcessor:
    def __init__(
        self,
        *,
        l1b_source,
        l1b_data_vars,
        bbox=None,
        rp_sets: RetrackerProcessorSettings,
        retrack_sets: RetrackerSettings,
        fitting_sets: FittingSettings,
        sensor_sets: SensorSettings,
        wf_sets: WaveformSettings,
        nc_attrs_kw: dict = None,
        log_level: int = logging.INFO,
    ):
        simple_logger.set_root_logger(log_level=log_level)

        self.l1b_data_vars = l1b_data_vars
        self.rp_sets = RetrackerProcessorSettings() if rp_sets is None else rp_sets
        self.l1b_src_files, self.l2_dest_files = get_nc_src_dest_file_list(
            l1b_source, self.rp_sets.nc_dest_dir
        )
        self.bbox = (
            bbox if bbox is not None else [None for i in range(len(self.l1b_src_files))]
        )

        logging.info(
            f"Started retracking processing of {len(self.l1b_src_files)} L1B files!"
            + f"(n_cores={os.cpu_count() if rp_sets.n_procs is None else rp_sets.n_procs})\n"
            + (
                ", ".join([f"{k}: {v}" for k, v in nc_attrs_kw.items()])
                if nc_attrs_kw is not None
                else ""
            )
        )

        self.retrack_sets = (
            RetrackerSettings(settings_preset=SettingsPreset.NONE)
            if retrack_sets is None
            else retrack_sets
        )
        self.fitting_sets = FittingSettings() if fitting_sets is None else fitting_sets
        self.sensor_sets = sensor_sets if sensor_sets is not None else SensorSettings()
        self.wf_sets = (
            wf_sets
            if wf_sets is not None
            else WaveformSettings.get_default_src_type(L1bSourceType.EUM_S3)
        )
        self.nc_attrs_kw = nc_attrs_kw if nc_attrs_kw is not None else {}

        self.retracker = SamosaRetracker(
            retrack_sets=self.retrack_sets,
            fitting_sets=self.fitting_sets,
            sensor_sets=self.sensor_sets,
            wf_sets=self.wf_sets,
        )

        if self.rp_sets.do_create_settings_log_file:
            self.write_settings_log_file()

    def get_init_output_l2(self, *, reduce_factor=1, reduce_offset=0):
        if reduce_factor > 1:
            unique_vals, inds = np.unique(
                self.l1b_data["record_inds"], return_index=True
            )
            inds += reduce_offset
        else:
            # assumes that we process all entries in l1b_data
            inds = np.arange(len(self.l1b_data["record_inds"]))
        n_inds = len(inds)

        # pre-allocate output Dataset
        return xr.Dataset.from_dict(
            {
                "data_vars": {
                    "swh": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {"units": "m"},
                    },
                    "swh_qual": {
                        "data": np.ones(n_inds),
                        "dims": "time",
                        "attrs": {
                            "units": "bool",
                            "long_name": "standard quality flag for the swh variable, "
                            '"misfit_selective" > 4 (if AIM of CORAL is activated,'
                            "otherwise it is simply the misfit)",
                            "flag_values": "[0 1]",
                            "flag_meaning": "good bad",
                        },
                    },
                    "swh_alt_qual": {
                        "data": np.ones(n_inds),
                        "dims": "time",
                        "attrs": {
                            "units": "bool",
                            "long_name": "(alternative) quality flag for the swh variable, "
                            '"misfit_le" > 4 (the misfit over the detected leading edge)',
                            "flag_values": "[0 1]",
                            "flag_meaning": "good bad",
                        },
                    },
                    "swh_alt2_qual": {
                        "data": np.ones(n_inds),
                        "dims": "time",
                        "attrs": {
                            "units": "bool",
                            "long_name": "second (alternative) quality flag for the swh variable, "
                            '"misfit" > 4 (the default misfit over all range gates)',
                            "flag_values": "[0 1]",
                            "flag_meaning": "good bad",
                        },
                    },
                    "epoch": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {
                            "units": "ns",
                            "long_name": "epoch relative from epoch reference gate",
                        },
                    },
                    "range": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {
                            "units": "m",
                            "long_name": "retracked range at 20 Hz (no geo-corrections applied)",
                        },
                    },
                    "range_qual": {
                        "data": np.ones(n_inds),
                        "dims": "time",
                        "attrs": {
                            "units": "bool",
                            "long_name": "standard quality flag for the range variable, "
                            '"misfit_selective" > 4 (if AIM of CORAL is activated,'
                            "otherwise it is simply the misfit)",
                            "flag_values": "[0 1]",
                            "flag_meaning": "good bad",
                        },
                    },
                    "range_alt_qual": {
                        "data": np.ones(n_inds),
                        "dims": "time",
                        "attrs": {
                            "units": "bool",
                            "long_name": "(alternative) quality flag for the range variable, "
                            '"misfit_le" > 4 (the misfit over the detected leading edge)',
                            "flag_values": "[0 1]",
                            "flag_meaning": "good bad",
                        },
                    },
                    "range_alt2_qual": {
                        "data": np.ones(n_inds),
                        "dims": "time",
                        "attrs": {
                            "units": "bool",
                            "long_name": "second (alternative) quality flag for the range variable, "
                            '"misfit" > 4 (the default misfit over all range gates)',
                            "flag_values": "[0 1]",
                            "flag_meaning": "good bad",
                        },
                    },
                    "altitude": {
                        "data": self.l1b_data["alt_m"][inds],
                        "dims": "time",
                        "attrs": {"units": "m", "long_name": "altitude of satellite"},
                    },
                    "tracker_range": {
                        "data": self.l1b_data["tracker_range_m"][inds],
                        "dims": "time",
                        "attrs": {
                            "units": "m",
                            "long_name": "range of tracker window related to the epoch reference gate (epoch_ref_gate). ",
                        },
                    },
                    "Pu": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {"units": "W"},
                    },
                    "misfit": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {
                            "long_name": "misfit between fitted and fitted model waveform according to formula $misfit = 100 * sqrt(1/N * sum(wf - wf_model)**2)$"
                        },
                    },
                    "misfit_le": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {
                            "long_name": "leading-edge misfit between fitted and fitted model waveform's leading edge according to formula $misfit = 100 * sqrt(1/N * sum(wf - wf_model)**2)$"
                        },
                    },
                    "misfit_selective": {
                        "data": np.full(n_inds, np.nan),
                        "dims": "time",
                        "attrs": {
                            "long_name": "selective (interference is excluded) misfit between fitted and fitted model waveform according to formula $misfit = 100 * sqrt(1/N * sum(wf - wf_model)**2)$"
                        },
                    },
                    "dist2coast": {
                        "data": self.l1b_data["dist2coast"][inds],
                        "dims": "time",
                        "attrs": {"units": "km"},
                    },
                    "record_ind": {
                        "data": self.l1b_data["record_inds"][inds],
                        "dims": "time",
                        "attrs": {
                            "units": "count",
                            "long_name": "absolute record index as taken from L1BS file",
                        },
                    },
                },
                "coords": {
                    "time": {"data": self.l1b_data["time"][inds], "dims": "time"},
                    "latitude": {
                        "data": np.degrees(self.l1b_data["lat_rad"][inds]),
                        "dims": "latitude",
                        "attrs": {"units": "degree"},
                    },
                    "longitude": {
                        "data": np.degrees(self.l1b_data["lon_rad"][inds]),
                        "dims": "longitude",
                        "attrs": {"units": "degree"},
                    },
                },
                "attrs": {
                    **{
                        "processor": "PySAMOSA L2 Retracker for Delay-Doppler Altimetry",
                        "processor version": __version__,
                        "epoch_ref_gate": int(self.l1b_data["epoch_ref_gate"]),
                    },
                    **self.nc_attrs_kw,
                },
            }
        )

    @classmethod
    def _process_single(cls, retracker_obj, l1b_data_single, model_params, abs_ind):
        logging.debug(f"Processing waveform #{abs_ind}")
        try:
            res_fit = retracker_obj.fit_wf(
                l1b_data_single=l1b_data_single, model_params=model_params
            )
        except (ValueError, RuntimeError) as e:
            logging.error(
                f'Error: Stopped retracking of waveform #{abs_ind}: {" ".join(e.args)}'
            )
            return {}

        logging.debug(
            f"Finished waveform #{abs_ind} (n_iter={res_fit['n_iter']}). SWH={res_fit['swh']:.2f}m,"
            f"epoch={res_fit['epoch_ns']:.2f}ns, Pu={res_fit['Pu']:.2f}W. "
        )

        return res_fit

    def _write_fit_results(self, rel_ind, res_fit):
        if res_fit:
            self.output_l2.swh[rel_ind] = res_fit["swh"]
            self.output_l2.swh_qual[rel_ind] = res_fit["misfit_selective"] > 4
            self.output_l2.swh_alt_qual[rel_ind] = res_fit["misfit_le"] > 4
            self.output_l2.swh_alt2_qual[rel_ind] = res_fit["misfit"] > 4

            self.output_l2.epoch[rel_ind] = res_fit["epoch_ns"]
            self.output_l2.range[rel_ind] = self.l1b_data["tracker_range_m"][
                rel_ind
            ] + (float(res_fit["epoch_ns"])) * 1e-9 * (CONST_C / 2)
            self.output_l2.range_qual[rel_ind] = self.output_l2.swh_qual[rel_ind]
            self.output_l2.range_alt_qual[rel_ind] = self.output_l2.swh_alt_qual[
                rel_ind
            ]
            self.output_l2.range_alt2_qual[rel_ind] = self.output_l2.swh_alt2_qual[
                rel_ind
            ]

            self.output_l2.Pu[rel_ind] = res_fit["Pu"]
            self.output_l2.misfit[rel_ind] = res_fit["misfit"]
            self.output_l2.misfit_le[rel_ind] = res_fit["misfit_le"]
            self.output_l2.misfit_selective[rel_ind] = res_fit["misfit_selective"]

    def process(self):
        for nc_src_file, nc_dest_filepath, bbox in zip(
            self.l1b_src_files, self.l2_dest_files, self.bbox
        ):
            try:
                nc_dest_dir = nc_dest_filepath.parent
                nc_base_id = (
                    nc_dest_filepath.parent.stem
                    if "measurement" in nc_dest_filepath.stem
                    else nc_dest_filepath.stem
                )

                if (
                    self.rp_sets.do_write_out_nc
                    and self.rp_sets.skip_if_exists
                    and nc_dest_filepath.exists()
                ):
                    logging.info(f"{nc_base_id} already exists, skipping...")
                    continue

                log_filename = nc_dest_dir / f"log_{nc_base_id}.txt"
                if self.rp_sets.do_write_out_log:
                    simple_logger.generate_and_add_file_logger(log_filename)

                logging.info(f"Reading of L1B dataset {nc_base_id}... ")
                grps = [
                    g for g in list(netCDF4.Dataset(nc_src_file).groups) if "data" in g
                ]
                nc_grp = f"{grps[0]}/ku" if grps else ""

                # if bbox is not None replace n_offset and n_inds by bbox logic
                if bbox is not None:
                    # hack required because some nc files have issues with
                    # decoding of times
                    try:
                        ds = xr.open_dataset(nc_src_file, group=nc_grp)
                    except BaseException:
                        ds = xr.open_dataset(
                            nc_src_file, group=nc_grp, decode_times=False
                        )

                    lat_deg = ds[self.l1b_data_vars["lat_rad"]].values
                    lon_deg = ds[self.l1b_data_vars["lon_rad"]].values

                    ds.close()

                    eps = 0.001
                    inds_in_bbox = sorted(
                        np.where(
                            (lat_deg > (bbox[0] - eps))
                            & (lat_deg < (bbox[1] + eps))
                            & (lon_deg > (bbox[2] - eps))
                            & (lon_deg < (bbox[3] + eps))
                        )[0]
                    )
                    self.rp_sets.n_offset = int(inds_in_bbox[0])
                    self.rp_sets.n_inds = len(inds_in_bbox)

                self.l1b_data = _read_dataset_vars_from_ds(
                    nc_filename=nc_src_file,
                    data_var_names=self.l1b_data_vars,
                    n_offset=self.rp_sets.n_offset,
                    n_inds=self.rp_sets.n_inds,
                    group=nc_grp,
                )

                logging.info("Finished reading.")

                if self.rp_sets.do_dynamic_fg_epoch:
                    logging.info("Calculating dynamic_first_guess_epochs... ")

                    fg_epoch = get_dynamic_first_guess_epochs(
                        wfs=self.l1b_data["wf"],
                        tracker_range=self.l1b_data["tracker_range_m"],
                        alt_m=self.l1b_data["alt_m"],
                        bu_bw_Hz=self.sensor_sets.B_r_Hz,
                        fg_epoch_adjacent_meas=self.rp_sets.dynamic_fg_epoch_n_adjacent_meas,
                        n_procs=self.rp_sets.n_procs,
                    )
                    self.l1b_data["dynamic_fg_epoch"] = fg_epoch

                    logging.info("Finished calculating dynamic_first_guess_epochs.")

                self.output_l2 = self.get_init_output_l2()

                # select indices to process: ocean waveforms only
                mask_inds_process = (self.output_l2.dist2coast >= 0).values
                abs_inds_process = (
                    self.rp_sets.n_offset + np.arange(self.output_l2.swh.shape[0])
                )[mask_inds_process]
                rel_inds_process = mask_inds_process.nonzero()[0]

                # Start processing
                logging.info(f"Started processing of L1B dataset {nc_base_id}... ")
                if self.rp_sets.n_procs == 1:
                    for rel_ind, abs_ind in zip(rel_inds_process, abs_inds_process):
                        res_fit = RetrackerProcessor._process_single(
                            retracker_obj=self.retracker,
                            l1b_data_single=get_subset_dataset(self.l1b_data, rel_ind),
                            model_params=get_model_param_obj_from_l1b_data(
                                self.l1b_data, rel_ind
                            ),
                            abs_ind=abs_ind,
                        )
                        self._write_fit_results(rel_ind, res_fit)
                else:
                    with futures.ProcessPoolExecutor(
                        max_workers=self.rp_sets.n_procs
                    ) as pool:
                        for rel_ind, res_fit in zip(
                            rel_inds_process,
                            pool.map(
                                RetrackerProcessor._process_single,
                                repeat(self.retracker),
                                gen_subset_dataset(self.l1b_data, rel_inds_process),
                                gen_model_param_obj_from_l1b_data(
                                    self.l1b_data, rel_inds_process
                                ),
                                abs_inds_process,
                                chunksize=20,
                            ),
                        ):
                            self._write_fit_results(rel_ind, res_fit)

                logging.info(f"Finished processing of L1B dataset {nc_base_id}. ")

                # Write netCDF file
                if self.rp_sets.do_write_out_nc:
                    os.umask(0)
                    nc_dest_dir.mkdir(parents=True, exist_ok=True)
                    self.reduce_l2()
                    self.output_l2.to_netcdf(nc_dest_filepath)

                    logging.info(f"netCDF file written. {nc_dest_filepath}")
            except RuntimeError as e:
                logging.error(
                    f"Error has occurred during processing of file {nc_src_file}: {e}"
                )
                # logging.root.removeHandler(file_logger_handle)
                logging.shutdown()

        logging.info(
            f"Processing done. ({len(self.l1b_src_files)} netCDF files. RetrackerProcessorSettings{self.rp_sets})"
        )

    def reduce_l2(self):
        if self.rp_sets.reduce_l2_factor > 1:
            reduce_offset = (
                self.rp_sets.reduce_l2_factor // 2
                if (self.rp_sets.reduce_l2_factor % 2) != 0
                else self.rp_sets.reduce_l2_factor // 2 - 1
            )
            ds_reduced = self.get_init_output_l2(
                reduce_factor=self.rp_sets.reduce_l2_factor, reduce_offset=reduce_offset
            )
            not self.retrack_sets.fit_zero_doppler
            len(self.output_l2.record_ind)

            for rel_ind, rec_ind in enumerate(np.unique(self.output_l2.record_ind)):
                mask_sel = self.output_l2.record_ind == rec_ind
                abs_mask_good_recs = mask_sel & ~(self.output_l2.swh_qual == 1)

                if all(~abs_mask_good_recs):  # if all are bad estimates
                    avg_func = np.nanmean
                    ds_reduced.swh[rel_ind] = avg_func(self.output_l2.swh[mask_sel])
                    ds_reduced.swh_qual[rel_ind] = 1.0
                    ds_reduced.epoch[rel_ind] = avg_func(self.output_l2.epoch[mask_sel])
                    ds_reduced.range[rel_ind] = avg_func(self.output_l2.range[mask_sel])
                    ds_reduced.Pu[rel_ind] = avg_func(self.output_l2.Pu[mask_sel])
                else:
                    avg_func = np.nanmean
                    ds_reduced.swh[rel_ind] = avg_func(
                        self.output_l2.swh[abs_mask_good_recs]
                    )
                    ds_reduced.swh_qual[rel_ind] = (
                        float(self.output_l2.swh_qual[abs_mask_good_recs])
                        if np.isscalar(self.output_l2.swh_qual[abs_mask_good_recs])
                        else 0.0
                    )
                    ds_reduced.epoch[rel_ind] = avg_func(
                        self.output_l2.epoch[abs_mask_good_recs]
                    )
                    ds_reduced.range[rel_ind] = avg_func(
                        self.output_l2.range[abs_mask_good_recs]
                    )
                    ds_reduced.Pu[rel_ind] = avg_func(
                        self.output_l2.Pu[abs_mask_good_recs]
                    )

            self.output_l2 = ds_reduced

            logging.info(
                f"processed L2 dataset reduced by factor {self.rp_sets.reduce_l2_factor}. "
            )

    def write_settings_log_file(self):
        all_sets = ExportSettings(
            rp_sets=self.rp_sets,
            retrack_sets=self.retrack_sets,
            fitting_sets=self.fitting_sets,
            sensor_sets=self.sensor_sets,
        )

        dest_filepath_json = self.rp_sets.nc_dest_dir / "settings.json"
        os.umask(0)
        dest_filepath_json.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_filepath_json, "w") as f:
            f.write(all_sets.model_dump_json(indent=4))
