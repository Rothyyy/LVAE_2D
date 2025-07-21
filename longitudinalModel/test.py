import torch
import numpy as np
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D
from longitudinalModel.project_encodings_for_training import project_encodings_for_training
import pandas as pd

from longitudinalModel.utils import produce_encodings_df
from nnModels.CVAE2D import CVAE2D
from nnModels.losses import longitudinal_loss, spatial_auto_encoder_loss, loss_bvae2, spatial_2D_auto_encoder_loss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.loading_image import open_npy


def test(model, data_loader, longitudinal_estimator=None,
         device='cuda' if torch.cuda.is_available() else 'cpu', spatial_loss=spatial_auto_encoder_loss):
    """
    Test a variational autoencoder. If longitudinal_estimator is not None then the model will be trained in order for
    its encoding to respect the mixed effect model described by the longitudinal_estimator. Just like in the paper:

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
    model.training = False
    model.eval()

    total_loss = []
    total_recon_loss, total_kl_loss, total_alignment_loss = 0.0, 0.0, 0.0
    nb_batches = 0
    with torch.no_grad():
        for data in data_loader:
            nb_batches += 1
            input_ = data[0].to(device).float()
            mu, logVar, reconstructed, encoded = model(input_)
            reconstruction_loss, kl_loss = spatial_loss(mu, logVar, reconstructed, input_)
            loss = reconstruction_loss + model.beta * kl_loss
            if longitudinal_estimator is not None:
                encodings_df = produce_encodings_df(model, data, device)
                timepoints_of_projection, predicted_latent_variables = project_encodings_for_training(encodings_df,
                                                                                                      longitudinal_estimator,
                                                                                                      )
                alignment_loss = longitudinal_loss(mu, torch.cat(([
                    torch.tensor(predicted_latent_variables[str(subject_id)]).float().to(device) for subject_id in
                    data[2]])))
                total_alignment_loss += alignment_loss.item()
                loss += model.gamma * alignment_loss

            total_recon_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss.append(float(loss.item()))

    print("Reconstruction loss =", total_recon_loss / nb_batches, ",Weighted kl loss =",
          total_kl_loss * model.beta / nb_batches,
          ",Weighted alignment loss =", total_alignment_loss * model.gamma / nb_batches)

    return sum(total_loss) / len(total_loss)


if __name__ == "__main__":
    easy_dataset = LongitudinalDataset2D('starmen_train_set.csv', read_image=open_npy)
    validation_dataset = LongitudinalDataset2D('starmen_validation_set.csv', read_image=open_npy, )
    model = CVAE2D(4)
    model.load_state_dict(
        torch.load(f"saved_models_2D/modelnormalementcamarche_5.0_100.0_4_200.pth", map_location='cpu'))
    validation_data_loader = DataLoader(validation_dataset, batch_size=256, num_workers=0, shuffle=False,
                                        collate_fn=longitudinal_collate_2D)

    test(model, validation_data_loader, longitudinal_estimator=None,
         device='cuda' if torch.cuda.is_available() else 'cpu', spatial_loss=spatial_2D_auto_encoder_loss)
