import torch
import numpy as np


def open_npy(path):
    return torch.from_numpy(np.load(path)).float()