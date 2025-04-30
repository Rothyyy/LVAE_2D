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
from dataset.group_based_train_test_split import group_based_train_test_split

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import spatial_auto_encoder_loss

from utils_display.display_individual_observations_2D import project_encodings_for_results
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D

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
    return encodings, logvars, projected_images

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="darker_circle")
    args = parser.parse_args()

    anomaly = args.anomaly
    anomaly_list = ["darker_circle", "darker_line", "growing_circle"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'darker_circle', 'darker_line', 'growing_circle' ")
        exit()

    latent_dimension = 4
    num_workers = round(os.cpu_count()/6)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading anomaly dataset
    anomaly_dataset_path = f"anomaly_{anomaly}_starmen_dataset.csv"
    dataset = Dataset2D(anomaly_dataset_path)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)


    # Loading VAE model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model_path = "saved_models_2D/CVAE2D_4_5_100_200.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)


    # Loading thresholds
    threshold_path = "anomaly_threshold.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)
    VAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    VAE_threshold_99 = threshold_dict["VAE_threshold_99"]

    num_images = 0
    VAE_anomaly_detected_95 = 0
    VAE_anomaly_detected_99 = 0
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            mu, logvar, recon_images, _ = model(images)
            reconstruction_loss, kl_loss = spatial_auto_encoder_loss(mu, logvar, recon_images, images)
            num_images += 1     # We load images one by one
            VAE_anomaly_detected_95 += torch.sum(reconstruction_loss > VAE_threshold_95).item()
            VAE_anomaly_detected_99 += torch.sum(reconstruction_loss > VAE_threshold_99).item()
            # print("VAE Reconstruction loss =", reconstruction_loss)
            # print("With threshold_95 :", reconstruction_loss > VAE_threshold_95)
            # print("With threshold_99 :", reconstruction_loss > VAE_threshold_99)
            # print()



    # Loading anomaly dataset
    transformations = transforms.Compose([])
    dataset = LongitudinalDataset2D(anomaly_dataset_path, read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D)

    # Loading LVAE model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model_path = "saved_models_2D/CVAE2D_4_5_100_200.pth2"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.training = False

    longitudinal_saving_path = "saved_models_2D/longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path)

    # Loading LVAE thresholds
    LVAE_threshold_95 = threshold_dict["LVAE_threshold_95"]
    LVAE_threshold_99 = threshold_dict["LVAE_threshold_99"]

    LVAE_anomaly_detected_95 = 0
    LVAE_anomaly_detected_99 = 0
    # TODO: How should we detect anomaly with longitudinal data ?
    with torch.no_grad():
        for data in data_loader:
            x = data[0]
            mus, logvars, recon_x = get_longitudinal_images(data, model, longitudinal_estimator)
            for i in range(len(mus)):
                reconstruction_loss, kl_loss = spatial_auto_encoder_loss(mus[i], logvars[i], recon_x[i], x[i])
                LVAE_anomaly_detected_95 += torch.sum(reconstruction_loss > VAE_threshold_95).item()
                LVAE_anomaly_detected_99 += torch.sum(reconstruction_loss > VAE_threshold_99).item()
                # print("LVAE Reconstruction loss =", reconstruction_loss)
                # print("With threshold_95 :", reconstruction_loss > LVAE_threshold_95)
                # print("With threshold_99 :", reconstruction_loss > LVAE_threshold_99)
                # print()
            
            # reconstruction_loss, kl_loss = spatial_auto_encoder_loss(mus, logvars, recon_x, x)
            # print("If we consider all images:")
            # print("LVAE Reconstruction loss =", reconstruction_loss)
            # print("With threshold_95 :", reconstruction_loss > LVAE_threshold_95)
            # print("With threshold_99 :", reconstruction_loss > LVAE_threshold_99)
            # print()



    print(f"With VAE and {num_images} images:")
    print(f"With threshold_95 we detect {VAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {VAE_anomaly_detected_99} anomaly.")
    print()

    print(f"With LVAE and {num_images} images:")
    print(f"With threshold_95 we detect {LVAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {LVAE_anomaly_detected_99} anomaly.")
    print()

