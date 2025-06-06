import numpy as np
import pandas as pd
import torch
from leaspy import AlgorithmSettings, Leaspy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import os

from dataset.Dataset2D import Dataset2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL

from utils_display.display_individual_observations_2D import display_individual_observations_2D
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D

from nnModels.losses import spatial_auto_encoder_loss


def CV_VAE(model_type, fold_index_list, test_set, nn_saving_path,
           device='cuda' if torch.cuda.is_available() else 'cpu',
           latent_dimension=4, gamma=100, beta=5,
           batch_size=256, num_worker=round(os.cpu_count()/4)):

    best_index = 0
    folds_test_loss = np.zeros(len(fold_index_list))

    for fold_index in range(len(fold_index_list)):
        model = model_type(latent_dimension)
        model.gamma=gamma
        model.beta=beta
        model.load_state_dict(torch.load(nn_saving_path, map_location='cpu'))

        model.device = device
        model.to(device)
        model.eval()
        model.training=False
        losses = []

        dataset = Dataset2D(test_set)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True, pin_memory=True)

        for x in data_loader:
            x = x.to(device)

            mu, logvar, recon_x, _ = model(x)
            reconstruction_loss, kl_loss = spatial_auto_encoder_loss(mu, logvar, recon_x, x)
            loss = reconstruction_loss + kl_loss * model.beta
            losses.append(loss)


        train_mean_loss = sum(losses) / len(losses)
        folds_test_loss[fold_index] = train_mean_loss

    return best_index

if __name__ == "__main__":
    print("Hello World")



