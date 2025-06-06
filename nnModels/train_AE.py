import torch
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torchvision.transforms as transforms
from dataset.Dataset2D import Dataset2D
from torch.utils.data import DataLoader

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

def train_AE_kfold(model, k_folds_index_list, nb_epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu',
             nn_saving_path=None, loss_graph_saving_path=None, spatial_loss=spatial_auto_encoder_loss, batch_size=256, num_workers=round(os.cpu_count()/4)):
    """
    Trains a variational autoencoder. Nothing longitudinal.
    The goal here is because an AE just requires image to train, it's easier to train and used already implemented
    techniques like pin_memory in the data loader.

    :args: model: variational autoencoder model to train
    :args: k_folds_dataset: list containing the numbers of folds in the dataset
    :args: nb_epochs: number of epochs for training
    :args: device: device used to do the variational autoencoder training
    """
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-8)
    model.to(device)
    model.device = device
    transformations = transforms.Compose([])
    losses = []
    best_val_loss = float('inf')
    iterator = tqdm(range(1, nb_epochs + 1), desc="Training", file=sys.stdout)

    folds_df_list = [pd.read_csv(f"data_csv/train_folds/starmen_train_set_fold_{i}.csv") for i in k_folds_index_list]
    nb_epochs_without_loss_improvement = 0
    valid_index = 0
    for valid_index in range(len(folds_df_list)):
        # Selecting validation and training dataframe
        valid_df = folds_df_list[valid_index]
        train_df = pd.concat([ folds_df_list[i] for i in range(len(folds_df_list)) if i != valid_index ], ignore_index=True)
        
        # Loading them in the Dataset2D class
        valid_dataset = Dataset2D(valid_df, transform=transformations)
        train_dataset = Dataset2D(train_df, transform=transformations)

        # Create the DataLoader
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

        for epoch in iterator:
            model.train()
            model.training = True
            train_loss = []

            # Training step
            for x in train_data_loader:
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

            # Validation step
            model.eval()
            model.training = False
            val_loss = []
            with torch.no_grad():
                for x in valid_data_loader:
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
                nb_epochs_without_loss_improvement = 0
                best_val_loss = val_mean_loss
                torch.save(model.state_dict(), f"fold_{valid_index}"+nn_saving_path)
            else:
                nb_epochs_without_loss_improvement += 1
            
            if nb_epochs_without_loss_improvement >= 30:
                break

            plt.plot(np.arange(1, len(losses) + 1), losses)
            if loss_graph_saving_path is not None:
                os.makedirs(os.path.dirname(loss_graph_saving_path), exist_ok=True)
                plt.savefig(loss_graph_saving_path + "fold_{valid_index}.pdf")
            plt.show()
        

    return losses, best_val_loss