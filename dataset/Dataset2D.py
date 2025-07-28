import pandas as pd
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.loading_image import open_npy


class Dataset2D(Dataset):
    """
    Dataset made to train the autoencoder without the longitudinal component on the synthetic starmen dataset.
    """

    def __init__(self, summary_file, transform=None, target_transform=None,
                 read_image=open_npy):      # read_image=lambda x: torch.Tensor(plt.imshow(x))
        if type(summary_file) == str:
            self.summary_dataframe = pd.read_csv(summary_file)
        elif type(summary_file) == pd.core.frame.DataFrame:
            self.summary_dataframe = summary_file
        else:
            print("Error type when loading data, check file name or type")
            exit()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32))
        ])

    def __len__(self):
        return len(self.summary_dataframe)

    def __getitem__(self, idx):
        summary_rows = self.summary_dataframe.iloc[idx]
        img_path = summary_rows['image_path']
        image = np.load(img_path)
        image = self.transform(image)
        return image
    
class Dataset2D_patch(Dataset):
    """
    Dataset made to train the autoencoder without the longitudinal component on the synthetic starmen dataset.
    """

    def __init__(self, summary_file, transform=None, target_transform=None,
                 read_image=open_npy):      # read_image=lambda x: torch.Tensor(plt.imshow(x))
        if type(summary_file) == str:
            self.summary_dataframe = pd.read_csv(summary_file)
        elif type(summary_file) == pd.core.frame.DataFrame:
            self.summary_dataframe = summary_file
        else:
            print("Error type when loading data, check file name or type")
            exit()
        self.transform = transform

    def __len__(self):
        return len(self.summary_dataframe)

    def __getitem__(self, idx):
        summary_rows = self.summary_dataframe.iloc[idx]
        patch_path = summary_rows['patch_path']
        patch = np.load(patch_path)
        patch = torch.from_numpy(patch).unsqueeze(1).float()

        if self.transform:
            patch = self.transform(patch)
            
        return patch


def collate_2D_patch(batch):
    images = torch.cat([patch for patch in batch if patch[0] is not None], axis=0)
    return images



if __name__ == "__main__":
    def open_npy(path):
        return torch.from_numpy(np.load(path).round()).float()


    dataset = Dataset2D('starmen_train_set.csv', read_image=open_npy,
                        transform=None)
    data_loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=True, pin_memory=True)
    data = next(iter(data_loader))
    print(data.size())
