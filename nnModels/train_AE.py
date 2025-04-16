import torch
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from longitudinalModel.test import test
from nnModels.losses import spatial_auto_encoder_loss, loss_bvae, loss_bvae2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def train_AE(model, data_loader, nb_epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu',
             nn_saving_path=None, loss_graph_saving_path=None, spatial_loss=spatial_auto_encoder_loss,
             validation_data_loader=None):
    """
    Trains a variational autoencoder. Nothing longitudinal.
    The goal here is because an AE just requires image to train, it's easier to train and used already implemented
    techniques like pin_memory in the data loader.

    :args: model: variational autoencoder model to train
    :args: data_loader: DataLoader to load the training data
    :args: nb_epochs: number of epochs for training
    :args: device: device used to do the variational autoencoder training
    """
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-8)
    model.to(device)
    model.device = device
    losses = []
    best_val_loss = float('inf')
    iterator = tqdm(range(1, nb_epochs + 1), desc="Training", file=sys.stdout)
    for epoch in iterator:
        model.train()
        model.training = True
        train_loss = []
        for x in data_loader:
            optimizer.zero_grad()
            x = x.to(device)

            mu, logvar, recon_x, _ = model(x)
            reconstruction_loss, kl_loss = spatial_loss(mu, logvar, recon_x, x)
            loss = reconstruction_loss + kl_loss * model.beta
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss.append(loss.item())

        train_mean_loss = sum(train_loss) / len(train_loss)
        val_mean_loss = train_mean_loss
        if validation_data_loader is not None:
            # Validation
            model.eval()
            model.training = False
            val_loss = []
            with torch.no_grad():
                for x in validation_data_loader:
                    x = x.to(device)
                    mu, logvar, recon_x, _ = model(x)
                    reconstruction_loss, kl_loss = spatial_loss(mu, logvar, recon_x, x)
                    loss = reconstruction_loss + kl_loss * model.beta
                    val_loss.append(loss.item())

            val_mean_loss = sum(val_loss) / len(val_loss)
        losses.append(val_mean_loss)
        iterator.set_postfix(
            {"Epoch": epoch, "Train mean loss": train_mean_loss, "Validation mean loss": val_mean_loss})
        print("\n")

        # Save model if validation loss decreased
        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            torch.save(model.state_dict(), nn_saving_path)
        plt.plot(np.arange(1, len(losses) + 1), losses)
        if loss_graph_saving_path is not None:
            os.makedirs(os.path.dirname(loss_graph_saving_path), exist_ok=True)
            plt.savefig(loss_graph_saving_path)
        plt.show()

    return losses, best_val_loss

