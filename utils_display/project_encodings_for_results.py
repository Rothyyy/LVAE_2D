from leaspy import AlgorithmSettings, Data


def project_encodings_for_results(data_df, subject_id, longitudinal_estimator, projection_timepoints,
                                  number_of_observations_as_base=3):
    """
    Gets the projection in time of encodings.
    arguments:
    @
    """
    print(data_df.head())
    origin_df = data_df.groupby('ID').apply(
        lambda x: x.sample(n=number_of_observations_as_base, replace=False)).reset_index(drop=True)
    origin_df.reset_index()
    print(origin_df.head())
    data = Data.from_dataframe(origin_df)
    print("projection_timepoints=",projection_timepoints)
    settings_personalization = AlgorithmSettings('scipy_minimize', use_jacobian=True)
    ip = longitudinal_estimator.personalize(data, settings_personalization)
    print(origin_df["ENCODING0"].tolist())
    print(ip[subject_id])
    reconstruction = longitudinal_estimator.estimate(projection_timepoints, ip)
    print(reconstruction)
    return reconstruction, origin_df['TIME'].tolist()
