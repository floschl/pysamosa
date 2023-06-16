samplusplus_test_inds = [
    510,
    2454,
    4980,
    12400,  # problematic with levmar fitting
    38500,  # normal wf
    40154,
    41023,
    12368,
]

# @pytest.mark.parametrize("n_offset", samplusplus_test_inds)
# def test_basic_retracking(n_offset, eumetsat_sam_l1bs, eumetsat_sam_l2):
#     fid = 's3_0'
#     l1b_data = eumetsat_sam_l1bs(n_offset=n_offset, n_inds=1, file_id=fid)
#     l1b_data_single = get_subset_dataset(l1b_data, ind_offset=0)
#     l2_data = eumetsat_sam_l2(n_offset=n_offset, n_inds=1, file_id=fid)
#     l2_data_single = get_subset_dataset(l2_data, ind_offset=0)
#     model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=0)
#
#     sensor_sets = SENSOR_SETS_DEFAULT_S3
#     simple_logger.set_root_logger()
#
#     wf_sets = WaveformSettings.get_default_src_type(L1bSourceType.GPOD)
#     sr = SamosaRetracker(
#         retrack_sets=RetrackerSettings(settings_preset=SettingsPreset.SAMPLUSPLUS),
#         fitting_sets=FittingSettings(),
#         sensor_sets=sensor_sets,
#         wf_sets=wf_sets
#     )
#
#     # start fitting
#     res_fit = sr.fit_wf(l1b_data_single=l1b_data_single, model_params=model_params)
#
#     # assert res_fit['misfit'] < 4
#
#     # RIP
#     ripa = RipAnalyser(l1b_data_single['rip'], sensor_sets=SENSOR_SETS_DEFAULT_S3, model_params=model_params)
#
#     # labelling
#     # fig, axs = plt.subplots(2, 1)
#     fig, axs = plt.subplots(2, 1, constrained_layout=True)
#     plot_retrack_result(l1b_data_single, l2_data_single, res_fit, model_params, sensor_sets=sensor_sets, model_sets=sr.model_sets, ax=axs[0], show_l2_model=False, wf_sets=wf_sets)
#     plot_rip_result(l1b_data_single, ripa, ripa.rip_params, ax=axs[1])
#
#     title = f'{eumetsat_sam_l1bs.get_nc_filename(file_id=fid).name} record#: {n_offset}'
#     fig.suptitle(title[:len(title)//2] + '\n' + title[len(title)//2:], fontsize=8)
#
#     fig.show()
#
#
# @pytest.mark.parametrize("n_offset", samplusplus_test_inds)
# def test_rip_analyser(n_offset, eumetsat_sam_l1bs):
#     fid = 's3_0'
#     l1b_data = eumetsat_sam_l1bs(n_offset=n_offset, n_inds=1, file_id=fid)
#     l1b_data_single = get_subset_dataset(l1b_data, ind_offset=0)
#     # l2_data = _read_dataset_vars_from_ds(nc_filename=l2_file, data_var_names=data_vars_l2, n_offset=n_offset, n_inds=n_inds)
#     # l2_data_single = get_single_result_from_dataset(l2_data, ind=0)
#
#     model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=0)
#
#     ripa = RipAnalyser(l1b_data_single['rip'], sensor_sets=SENSOR_SETS_DEFAULT_S3, model_params=model_params)
#
#     # plotting
#     # waveform
#     fig, axs = plt.subplots(2,1)
#     axs[0].plot(l1b_data_single['wf'])
#
#     # RIP
#     plot_rip_result(l1b_data_single, ripa, ripa.rip_params, ax=axs[1])
#
#     fig.show()
#
#
# @pytest.mark.parametrize("n_offset", samplusplus_test_inds)
# def test_gen_2d_rip(n_offset, eumetsat_sam_l1bs):
#     fid = 's3_0'
#     l1b_data = eumetsat_sam_l1bs(n_offset=n_offset, n_inds=1, file_id=fid)
#     l1b_data_single = get_subset_dataset(l1b_data, ind_offset=0)
#     model_params = get_model_param_obj_from_l1b_data(l1b_data, ind=0)
#
#     ripa = RipAnalyser(l1b_data_single['rip'], sensor_sets=SENSOR_SETS_DEFAULT_S3, model_params=model_params)
#
#     n_eff = 53
#     rip_2d_gaussian = ripa.get_gamma0(n_looks_eff=n_eff, n_gates=512)
#
#     # plots
#     fig, axs = plt.subplots(2, 2)
#     axs = axs.ravel()
#     fontsize_labels = 7
#
#     legend_kwargs_default = {'loc': 'upper left',
#                              **{'prop': {'size': fontsize_labels}, 'labelspacing': 0.50, 'borderaxespad': 0.5}}
#
#     # along-track RIP
#     axs[0].set_title('RIP waveform and fitted gaussian', fontsize=fontsize_labels)
#     axs[0].plot(ripa.doppler_beam_inds, ripa.rip_wf_norm, label='measured RIP_az')
#     axs[0].plot(ripa.doppler_beam_inds, ripa.rip_params.rip_az_fitted, '+', label='fitted RIP_az')
#     axs[0].legend(**legend_kwargs_default)
#     axs[0].axvline(0, color='red', linestyle='dashed', linewidth=0.5)
#     axs[0].axvline(ripa.looks_eff[-1], color='red', linestyle='dashed', linewidth=0.5)
#
#     axs[1].set_title('interpolated RIP_act', fontsize=fontsize_labels)
#     axs[1].plot(ripa.rip_act_eff)
#     axs[1].legend()
#
#     axs[2].set_title(f'RIP_az_oversampled, factor={ripa.oversampling_factor}', fontsize=fontsize_labels)
#     axs[2].plot(ripa.rip_wf_oversampled)
#     axs[2].plot(ripa.rip_az_fitted, linestyle='dashed')
#
#     axs[3].set_title('GAMMA0/2D RIP', fontsize=fontsize_labels)
#     axs[3].imshow(rip_2d_gaussian, cmap='jet', interpolation='nearest', aspect='auto');
#
#     # axs[3].set_title('2D RIP/GAMMA0 (measured RIP)')
#     # axs[3].imshow(rip_2d_exact, cmap='jet', interpolation='nearest', aspect='auto');
#
#     fig.show()
