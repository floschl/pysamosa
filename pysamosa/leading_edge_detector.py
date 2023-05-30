import numpy as np

from pysamosa.model import get_region_max


def detect_leading_edge(
    wf, fg_epoch=None, normalise_wf_by_fg_region=None, append_first_falling_edge=False
):
    """Detects the leading edge in a power return echo waveform

    :raises ValueError: wrong input arguments
    :raises RuntimeError: Leading edge detection failed
    :param wf: the waveform in samples
    :param fg_epoch: the first-guess epoch, if None is given the maximum of the waveform is chosen
    :param normalise_wf_by_fg_region: number of samples in gates around the fg_epoch the waveform is normalised
    :return: the indices of the gates for which the leading edge is detected, type np.ndarray
    """
    if fg_epoch is not None and normalise_wf_by_fg_region is None:
        raise ValueError("Error in leading edge detection. ")

    if fg_epoch and normalise_wf_by_fg_region:
        wf = wf / get_region_max(
            wf=wf, region_center=fg_epoch, n_before_after=normalise_wf_by_fg_region
        )
    else:
        wf = wf / np.max(wf)

    fg_epoch = np.argmax(wf) if fg_epoch is None else fg_epoch

    # get the numerical gradient of the waveform and all rising edges above a
    # certain threshold
    def get_rising_falling_edge(_wf, falling_edges=False):
        d = np.diff(_wf, prepend=[0])
        le_ind_mask = (d < 0.01) if falling_edges else (d > 0.01)

        # get all consecutive rising edges
        max_stepsize = 1
        data = le_ind_mask.nonzero()[0]
        return np.split(data, np.where(np.diff(data) > max_stepsize)[0] + 1)

    # remove indices from edges that are after fg_epoch
    rising_edges = get_rising_falling_edge(wf)

    rising_edges_shortened = [
        list(r)
        for r in rising_edges
        if (fg_epoch - 2 * normalise_wf_by_fg_region)
        <= r[-1]
        <= (fg_epoch + normalise_wf_by_fg_region)
    ]
    rising_edges_shortened = list(
        filter(None, rising_edges_shortened)
    )  # remove empty list

    # selected_le_gates = sorted(rising_edges_left_fg_shortend, key=len,
    # reverse=True)[0]  # sort rising edges by width
    selected_le_gates = sorted(
        rising_edges_shortened, key=lambda e: wf[e[-1]] - wf[e[0]], reverse=True
    )  # sort rising edges by height

    # fill potential gaps in series
    if len(selected_le_gates) and len(selected_le_gates[0]):
        selected_le_gates = np.arange(
            selected_le_gates[0][0], selected_le_gates[0][-1] + 1
        )
    else:
        raise RuntimeError("Leading edge could not be detected. ")

    # detect trailing edge after leading edge
    if append_first_falling_edge:
        falling_edges = get_rising_falling_edge(wf, falling_edges=True)
        falling_edge_after_le = [
            fe for fe in falling_edges if (fe[0] - 1) == selected_le_gates[-1]
        ][0]

        selected_le_gates = np.concatenate((selected_le_gates, falling_edge_after_le))

    return selected_le_gates


# class LeadingEdgeDetector():
#
#     def __init__(self):
#         self.init = True
#
#     def __call__(self, *args, **kwargs):
#         wf =
