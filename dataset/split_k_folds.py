import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

def train_k_folds_split(df_train, n_split=8) -> list:
    # Get unique subject IDs
    kfold = KFold(n_splits=n_split, shuffle=False)
    folds = list(kfold.split(range(len(df_train))))
    os.makedirs("data_csv/train_folds", exist_ok=True)
    for i in range(n_split):
        df_train.loc[folds[i][1]].to_csv(f"data_csv/train_folds/starmen_train_set_fold_{i}.csv", index=False)

    return [i for i in range(n_split)]

# Usage example
# train_df, test_df = group_based_train_test_split(df, test_size=0.2, group_col='subject_id', random_state=42)

# To check the result
# print(f"Original df shape: {df.shape}")
# print(f"Train df shape: {train_df.shape}")
# print(f"Test df shape: {test_df.shape}")
# print(f"Number of unique subjects in original df: {df['subject_id'].nunique()}")
# print(f"Number of unique subjects in train df: {train_df['subject_id'].nunique()}")
# print(f"Number of unique subjects in test df: {test_df['subject_id'].nunique()}")