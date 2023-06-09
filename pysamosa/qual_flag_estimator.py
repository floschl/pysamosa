import numpy as np
from scipy import signal

from pysamosa.retracker_helpers import get_inds_sliding_window


class SwhQualFlagEstimator:
    def __init__(self, swh: np.ndarray, ssh: np.ndarray):
        self.swh = np.asarray(swh)
        self.ssh = np.asarray(ssh)

        # init qual_flag with nan values (1.0)
        self.swh_qual = np.full(swh.shape, np.nan)
        self.swh_qual[np.isnan(self.swh) | np.isnan(self.ssh)] = 1.0

    def estimate_swh_qual_flag(
        self, n_window_size: int = 100, sigma_boundary: int = 1.5
    ):
        """Estimates a new swh qual flag based on a smart algortihm

        :param swh: significant wave height
        :param ssh: sea surface height
        :param n_window_size: window size over which the window is shifted over
        :return: the new q
        """

        for i, (inds, center_sample) in enumerate(
            get_inds_sliding_window(self.swh.shape[0], n=n_window_size)
        ):
            ssh_seg = self.ssh[inds]
            self.swh[inds]

            non_nans = (~np.isnan(ssh_seg)).nonzero()[0]
            if center_sample in non_nans:
                ssh_detrended = signal.detrend(ssh_seg[non_nans])
                sigma = np.std(ssh_detrended)

                # mark good inds in qual vec
                center_sample_wo_nans = int((non_nans == center_sample).nonzero()[0])
                if np.abs(ssh_detrended[center_sample_wo_nans]) < (
                    sigma_boundary * sigma
                ):
                    self.swh_qual[i] = 0.0
                else:
                    self.swh_qual[i] = 1.0

        return self.swh_qual
