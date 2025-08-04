import pandas as pd
import numpy as np
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", "-p", type=str, required=False, default=False)
    args = parser.parse_args()
    
    if args.patch in ["True", "y"]:
        print("IN patch split")
        dataset_path = "data_csv/starmen_dataset_patch.csv"
        dataset_df = pd.read_csv(dataset_path)
        train_df, test_df = group_based_train_test_split(dataset_df, test_size=0.2, group_col='subject_id', random_state=42)
        train_df.to_csv('data_csv/starmen_patch_train_set.csv', index=False)
        test_df.to_csv('data_csv/starmen_patch_test_set.csv', index=False)
    else:
        dataset_path = "data_csv/starmen_dataset.csv"
        dataset_df = pd.read_csv(dataset_path)
        train_df, test_df = group_based_train_test_split(dataset_df, test_size=0.2, group_col='subject_id', random_state=42)
        train_df.to_csv('data_csv/starmen_train_set.csv', index=False)
        test_df.to_csv('data_csv/starmen_test_set.csv', index=False)

# To check the result
# print(f"Original df shape: {df.shape}")
# print(f"Train df shape: {train_df.shape}")
# print(f"Test df shape: {test_df.shape}")
# print(f"Number of unique subjects in original df: {df['subject_id'].nunique()}")
# print(f"Number of unique subjects in train df: {train_df['subject_id'].nunique()}")
# print(f"Number of unique subjects in test df: {test_df['subject_id'].nunique()}")