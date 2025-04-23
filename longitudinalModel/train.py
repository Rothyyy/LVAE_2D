import torch
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from longitudinalModel.project_encodings_for_training import project_encodings_for_training
from longitudinalModel.test import test
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from nnModels.losses import longitudinal_loss, spatial_auto_encoder_loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F


def is_reconstruction_well_ordered(times, subject_ids, reconstruction_dict):
    for i in range(len(subject_ids)):
        for t in range(len(times[i])):
            if times[i][t] != reconstruction_dict[subject_ids[i]][t]:
                return False
    return True


def train(model, data_loader, longitudinal_estimator=None,
          longitudinal_estimator_settings=None, nb_epochs=100, lr=0.01,
          device='cuda' if torch.cuda.is_available() else 'cpu', nn_saving_path=None, longitudinal_saving_path=None,
          loss_graph_saving_path=None, previous_best_loss=1e15, spatial_loss=spatial_auto_encoder_loss,
          validation_data_loader=None):
    """
    Trains a variational autoencoder on a longitudinal dataset. If longitudinal_estimator is not None then the model
    will be trained in order for its encoding to respect the mixed effect model described by the longitudinal_estimator.
    Just like in the paper:

    :args: model: variational autoencoder model to train
    :args: data_loader: DataLoader to load the training data
    :args: latent_representation_size: number of dimension of the encodings
    :args: longitudinal_estimator: longitudinal mixed model to train
    :args: longitudinal_estimator_settings: training setting of the longitudinal model
    :args: encoding_csv_path: encodings for each observation stored in a CSV (then no need to do it at the beginning of
    the training
    :args: nb_epochs: number of epochs for training
    :args: lr: learning rate of the neural network model
    :args: device: device used to do the variational autoencoder training
    """
    model.to(device)
    model.device = device
    best_loss = previous_best_loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-5)
    nb_epochs_without_loss_improvement = 0
    losses = []

    iterator = tqdm(range(nb_epochs), desc="Training", file=sys.stdout)
    for epoch in iterator:
        nb_batch = 0
        model.training = True   # Unnecessary with the following line
        model.train()
        total_loss = []
        total_recon_loss, total_kl_loss, total_alignment_loss = 0.0, 0.0, 0.0
        ### Fit the longitudinal mixed effect model
        predicted_latent_variables = None
        timepoints_of_projection = None
        if longitudinal_estimator is not None:
            longitudinal_estimator, encodings_df = fit_longitudinal_estimator_on_nn(data_loader, model, device,
                                                                                    longitudinal_estimator,
                                                                                    longitudinal_estimator_settings)
            timepoints_of_projection, predicted_latent_variables = project_encodings_for_training(encodings_df,
                                                                                                  longitudinal_estimator,
                                                                                                  )


        for data in data_loader:
            nb_batch += 1
            optimizer.zero_grad()
            x = data[0].to(device).float()
            mu, logVar, reconstructed, encoded = model(x)
            reconstruction_loss, kl_loss = spatial_loss(mu, logVar, reconstructed, x)

            loss = reconstruction_loss + model.beta * kl_loss
            if longitudinal_estimator is not None:
                alignment_loss = longitudinal_loss(mu, torch.cat(([
                    torch.tensor(predicted_latent_variables[str(subject_id)]).float().to(device) for subject_id in
                    data[2]])))
                # if not is_reconstruction_well_ordered(times=data[1], subject_ids=data[2],
                #                                       reconstruction_dict=timepoints_of_projection):
                #     print('There is a problem in the reconstruction order')
                # are we sure that the order of the visits is the same in predicted latent variables
                # and in this data loader ??
                # i.e. maybe times in data[2] is [12,15,18] while in predicted_latent_variables the encodings
                # are ordered like this: [15,12,18], hence why i needed the reconstruction dict
                # loss = reconstruction_loss + model.beta * kl_loss + model.gamma * alignment_loss
                loss += model.gamma * alignment_loss
                total_alignment_loss += alignment_loss.item()


            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss.append(loss.item())
            total_recon_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
        print("\n Reconstruction loss =", total_recon_loss / nb_batch, ", Weighted KL loss =",
              total_kl_loss / nb_batch * model.beta,
              ", Weighted alignment loss =", total_alignment_loss / nb_batch * model.gamma, "\n")

        train_loss = sum(total_loss) / nb_batch
        epoch_loss = train_loss

        if validation_data_loader is not None:
            epoch_loss = test(model, validation_data_loader,
                              longitudinal_estimator=longitudinal_estimator,
                              device=device,
                              spatial_loss=spatial_loss)

        losses.append(epoch_loss)

        iterator.set_postfix({"epoch": epoch, "train loss": train_loss, "validation loss": epoch_loss, })

        if epoch_loss < best_loss:
            nb_epochs_without_loss_improvement = 0
            best_loss = epoch_loss
            if nn_saving_path is not None or longitudinal_saving_path is not None:
                print({"\n saving params..... \n"})
                if nn_saving_path is not None:
                    torch.save(model.state_dict(), nn_saving_path)
                if longitudinal_estimator is not None and longitudinal_saving_path is not None:
                    longitudinal_estimator.save(longitudinal_saving_path)
        else:
            nb_epochs_without_loss_improvement += 1

        if nb_epochs_without_loss_improvement >= 30:
            break
    print("\n")
    # plt.plot(np.arange(1, len(losses) + 1), losses, label="Loss")
    # if loss_graph_saving_path is not None:
    #     os.makedirs(os.path.dirname(loss_graph_saving_path), exist_ok=True)
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(loss_graph_saving_path)
    # plt.show()

    return best_loss, losses
