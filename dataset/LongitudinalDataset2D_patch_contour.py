import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils.loading_image import open_npy


class LongitudinalDataset2D_patch_contour(Dataset):
    # TODO: add to readme how to format the csv file
    def __init__(self, summary_file, transform=None, target_transform=None,
                 read_image=open_npy):     # read_image=lambda x: torch.Tensor(plt.imshow(x))
        if type(summary_file) == str:
            self.summary_dataframe = pd.read_csv(summary_file).sort_values(['subject_id', 'age'])
        elif type(summary_file) == pd.core.frame.DataFrame:
            self.summary_dataframe = summary_file.sort_values(['subject_id', 'age'])
        else:
            print("Error type when loading data, check file name or type")
            exit()
        self.transform = transform
        self.target_transform = target_transform
        self.read_image = read_image
        self.list_patient_ids = self.summary_dataframe['subject_id'].unique().tolist()

    def __len__(self):
        return len(self.list_patient_ids)

    def __getitem__(self, idx):
        """
        returns observations for an individual, the time of observation,the id of the individual
        images.shape = number_of_observations x 1 x Depth x Height x Width
        """
        patient_id = self.list_patient_ids[idx]
        summary_rows = self.summary_dataframe[self.summary_dataframe['subject_id'] == patient_id].sort_values(
            ['subject_id', 'age'])
        patches = [self.read_image(summary_rows.iloc[i]['patch_path']) for i in range(len(summary_rows))]
        patches = torch.cat(patches, dim=0)
        centers = [np.load(summary_rows.iloc[i]['centers_path']) for i in range(len(summary_rows))]
        centers = np.concatenate(centers)
        if len(patches) == 0:
            return None, None, None, None

        if self.transform:
            patches = self.transform(patches)

        # TODO: find a better way to code this, so there's no need to read each image 2 times
        return patches, [summary_rows['age'].iloc[i] for i in range(len(summary_rows))], patient_id, centers
    
    def get_patches_from_id(self, subject_id):
        """
        returns patches for an individual, the center of each patch, the time of observation
        patches.shape = number_of_observations x 1 x Depth x Height x Width
        """
        if subject_id not in self.list_patient_ids:
            print(f"Error subject {subject_id} not in dataset")
            return None, None, None, None
        summary_rows = self.summary_dataframe[self.summary_dataframe['subject_id'] == subject_id].sort_values(
            ['subject_id', 'age'])
        patches = [self.read_image(summary_rows.iloc[i]['patch_path']) for i in range(len(summary_rows))]
        patches = torch.cat(patches, dim=0)
        centers = [np.load(summary_rows.iloc[i]['centers_path']) for i in range(len(summary_rows))]
        centers = np.concatenate(centers)
        if len(patches) == 0:
            return None, None, None, None

        if self.transform:
            patches = self.transform(patches)

        return patches, [summary_rows['age'].iloc[i] for i in range(len(summary_rows))], centers


def longitudinal_collate_2D_patch_contour(batch, device='cuda' if torch.cuda.is_available() else 'cpu'):
    images = torch.cat([item[0] for item in batch if item[0] is not None], axis=0)
    # TODO: here use reshape instead of unsqueeze so the input size is fixed
    images = images.unsqueeze(1)
    infos = [item[1] for item in batch if item[0] is not None]
    # images = images.reshape(len(infos),1,)
    subject_ids = [item[2] for item in batch if item[0] is not None]
    centers = [item[3] for item in batch if item[0] is not None]

    return [images, infos, subject_ids, centers]
