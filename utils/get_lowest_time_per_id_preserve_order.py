def get_lowest_time_per_id_preserve_order(df):
    # Create a boolean mask for rows with minimum TIME per ID
    mask = df.groupby('ID')['TIME'].transform(min) == df['TIME']

    # Use the mask to filter the original dataframe
    result = df[mask]

    return result
