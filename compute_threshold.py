import numpy as np
import pandas as pd
import torch
from leaspy import AlgorithmSettings, Leaspy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import os
import json

from dataset.Dataset2D import Dataset2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import spatial_auto_encoder_loss, pixel_reconstruction_error

from utils_display.display_individual_observations_2D import project_encodings_for_results
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D



transformations = transforms.Compose([])


def open_npy(path):
    return torch.from_numpy(np.load(path)).float()


def get_longitudinal_images(data, model, fitted_longitudinal_estimator):
    encodings = []
    times = []
    ids = []
    subject_id = data[2][0]

    encoder_output = model.encoder(data[0].float().to(device))
    logvars = encoder_output[1].detach().clone().to(device)
    encodings.append(encoder_output[0].detach().clone().to(device))
    for i in range(len(data[1])):
        times.extend(data[1][i])
        ids.extend([data[2][i]] * len(data[1][i]))
    encodings = torch.cat(encodings)
    encodings_df = pd.DataFrame({'ID': ids, 'TIME': times})
    for i in range(encodings.shape[1]):
        encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                            encodings[:, i].detach().clone().tolist())
    encodings_df['ID'] = encodings_df['ID'].astype(str)
    projection_timepoints = {str(subject_id): data[1][0]}
    predicted_latent_variables, _ = project_encodings_for_results(encodings_df, str(subject_id),
                                                                                fitted_longitudinal_estimator,
                                                                                projection_timepoints)
    projected_images = model.decoder(torch.tensor(predicted_latent_variables[str(subject_id)]).to(device))
    # return encodings, logvars, projected_images
    
    return torch.from_numpy(predicted_latent_variables[str(subject_id)]), logvars, projected_images



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss_input", type=str, required=False, default="pixel")
    args = parser.parse_args()

    loss_input = args.loss_input
    if loss_input == "image":
        loss_function = spatial_auto_encoder_loss
    else:
        loss_function = pixel_reconstruction_error

    # Loading the VAE model
    latent_dimension = 4
    num_worker = round(os.cpu_count()/6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nn_saving_path = f"saved_models_2D/CVAE2D_4_5_100_200.pth"
    longitudinal_saving_path = f"saved_models_2D/longitudinal_estimator_params_CVAE2D_4_5_100_200.json"

    model = CVAE2D_ORIGINAL(latent_dimension)
    model.load_state_dict(torch.load(nn_saving_path, map_location='cpu'))
    model.to(device)
    model.training = False

    train_dataset = Dataset2D('data_csv/starmen_train_set.csv', read_image=open_npy,transform=transformations)
    data_loader = DataLoader(train_dataset, batch_size=1, num_workers=num_worker, shuffle=True, pin_memory=True, )
    all_losses = []

    # 1 epoch to get all reconstruction error with VAE
    with torch.no_grad():
        model.eval()
        for x in data_loader:
            x = x.to(device)

            mu, logvar, recon_x, _ = model(x)
            reconstruction_loss, kl_loss = loss_function(mu, logvar, recon_x, x)

            if loss_input == "pixel":
                reconstruction_loss = reconstruction_loss.numpy()
            loss = reconstruction_loss
            all_losses.append(loss)
    
    all_losses = np.array(all_losses).flatten()
    VAE_threshold_95 = np.percentile(all_losses, 95)
    VAE_threshold_99 = np.percentile(all_losses, 99)
    VAE_median = np.median(all_losses)
    VAE_losses_min = all_losses.min()
    VAE_losses_max = all_losses.max()
    VAE_losses_mean = all_losses.mean()




    # Loading the longitudinal model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model.load_state_dict(torch.load(nn_saving_path + "2", map_location='cpu'))
    model.to(device)
    model.training = False
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path + "2")


    train_dataset = LongitudinalDataset2D('data_csv/starmen_train_set.csv', read_image=open_npy, transform=transformations)
    data_loader = DataLoader(train_dataset, batch_size=1, num_workers=num_worker, shuffle=False, collate_fn=longitudinal_collate_2D)
    all_losses = []

    # 1 epoch to get all reconstruction error with LVAE
    with torch.no_grad():
        for data in data_loader:
            x = data[0]
            mus, logvars, recon_x = get_longitudinal_images(data, model, longitudinal_estimator)
            for i in range(len(mus)):
                reconstruction_loss, kl_loss = loss_function(mus[i], logvars[i], recon_x[i], x[i])
                if loss_input == "pixel":
                    reconstruction_loss = reconstruction_loss.numpy()
                all_losses.append(reconstruction_loss)


    all_losses = np.array(all_losses).flatten()
    LVAE_threshold_90 = np.percentile(all_losses, 90)
    LVAE_threshold_95 = np.percentile(all_losses, 95)
    LVAE_threshold_99 = np.percentile(all_losses, 99)
    LVAE_median = np.median(all_losses)
    LVAE_losses_min = all_losses.min()
    LVAE_losses_max = all_losses.max()
    LVAE_losses_mean = all_losses.mean()


    # Printing some stats
    print()
    print("Stats for VAE losses :")
    print("min =", VAE_losses_min)
    print("max =", VAE_losses_max)
    print("mean =", VAE_losses_mean)
    print("median =", VAE_median)
    print("95th percentile =", VAE_threshold_95)
    print("99th percentile =", VAE_threshold_99)

    print()
     
    print("Stats for LVAE losses :")
    print("min =", LVAE_losses_min)
    print("max =", LVAE_losses_max)
    print("mean =", LVAE_losses_mean)
    print("median =", LVAE_median)
    print(f"Number of {loss_input} above VAE_95 =", np.sum(all_losses > VAE_threshold_95))
    print("90th percentile =", LVAE_threshold_90)
    print("95th percentile =", LVAE_threshold_95)
    print("99th percentile =", LVAE_threshold_99)


    # Saving the obtained threshold
    threshold_dict = {}
    threshold_dict["VAE_threshold_95"] = VAE_threshold_95
    threshold_dict["VAE_threshold_99"] = VAE_threshold_99
    threshold_dict["LVAE_threshold_95"] = LVAE_threshold_95
    threshold_dict["LVAE_threshold_99"] = LVAE_threshold_99


    with open(f'data_csv/anomaly_threshold_{loss_input}.json', 'w') as f:
        json.dump(threshold_dict, f, ensure_ascii=False)
