import matplotlib.pyplot as plt

from pysamosa.common_types import L1bSourceType, RetrackerBaseType
from pysamosa.settings_manager import SettingsPreset, get_default_base_settings

test_inds_gpod = [
    # coast
    # 12327,
    # 12330,
    # 12330,
    # 12330,
    # *[v for v in range(12330, 12340)],
    # 12339,
    # 12340,
    # 12341,
    # *[v for v in range(12340, 12360)],
    # 12343,
    # open-ocean
    # *list(range(41658, 41668)),
    42354,
]


# rp_sets, retrack_sets, fitting_sets = get_default_base_settings(RetrackerBaseType.SAMPLUS, SettingsPreset.NONE)
rp_sets, retrack_sets, _, _, _ = get_default_base_settings(
    retracker_basetype=RetrackerBaseType.SAMPLUS,
    settings_preset=SettingsPreset.SAM_FLO,
    l1b_src_type=L1bSourceType.EUM_S3,
)

# rp_sets.do_dynamic_fg_epoch = True
rp_sets.dynamic_fg_epoch_n_adjacent_meas = 30  # default: 20

# retrack_sets.n_effective_looks = 0
# retrack_sets.normalise_wf_by_fg_region = 5
# retrack_sets.interference_masking = True
# retrack_sets.interference_masking_grow = 0
# retrack_sets.interference_masking_swh_max = 8.0
# retrack_sets.second_retracking_step_samplus = True

# fitting_sets.Levmar_Control_2 = 1e-1
# fitting_sets.lock_epoch_around_fg_n = True


def plot_leading_edge_analysis(le_inds, l1b_data_single, ind_gpod, fg_epoch):
    # plotting
    fig, axs = plt.subplots(1, 1)
    axs.plot(l1b_data_single["wf"].T)

    # plot initial first-guess epoch
    fg_epoch_style = {
        "linestyle": "--",
        "linewidth": 1,
        "color": "grey",
        "label": "fg_epoch",
    }
    axs.axvline(l1b_data_single["dynamic_fg_epoch"], **fg_epoch_style)

    # plot leading edge area
    axs.axvspan(le_inds[0], le_inds[-1], alpha=0.5, color="orange")
    axs.grid()
    axs.set_title(
        f"ind_gpod={ind_gpod}, len(le_inds)={len(le_inds)}, fg_epoch={fg_epoch}",
        fontsize=6,
    )

    fig.show()
