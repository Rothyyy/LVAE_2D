import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def group_based_train_test_split(df, test_size=0.2, group_col='subject_id', random_state=42):
    # Get unique subject IDs
    unique_subjects = df[group_col].unique()

    # Split the subject IDs into train and test groups
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state
    )

    # Create train and test dataframes based on the split subject IDs
    train_df = df[df[group_col].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_subjects)].reset_index(drop=True)

    return train_df, test_df

# Usage example
# train_df, test_df = group_based_train_test_split(df, test_size=0.2, group_col='subject_id', random_state=42)

# To check the result
# print(f"Original df shape: {df.shape}")
# print(f"Train df shape: {train_df.shape}")
# print(f"Test df shape: {test_df.shape}")
# print(f"Number of unique subjects in original df: {df['subject_id'].nunique()}")
# print(f"Number of unique subjects in train df: {train_df['subject_id'].nunique()}")
# print(f"Number of unique subjects in test df: {test_df['subject_id'].nunique()}")