from leaspy import AlgorithmSettings, Data


def project_encodings_for_training(data_df, longitudinal_estimator):
    """
    used for training. For a given encodings dataframe takes the first observation encodings for each patient and
    projects it at the time of the other observations.
    """
    # TODO: Check if we fit on first or on all the observations
    # origin_df = get_lowest_time_per_id_preserve_order(data_df.sort_values(['ID', 'TIME']))
    # maybe try a random projection instead of the first one
    data = Data.from_dataframe(data_df)
    settings_personalization = AlgorithmSettings('scipy_minimize', use_jacobian=True)
    ip = longitudinal_estimator.personalize(data, settings_personalization)  # TODO: Maybe change with sampling instead
                                                                             #       of parameters chosen to maximize likelihood
    reconstruction_dict = {}
    for i in range(len(data_df['ID'].unique())):
        subject_id = data_df['ID'].unique()[i]
        timepoints = data_df[data_df['ID'] == subject_id]['TIME'].tolist()
        timepoints.sort()
        reconstruction_dict[subject_id] = timepoints
    reconstruction = longitudinal_estimator.estimate(reconstruction_dict, ip)

    return reconstruction_dict, reconstruction
