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
import matplotlib.pyplot as plt

from dataset.Dataset2D import Dataset2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from dataset.group_based_train_test_split import group_based_train_test_split

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import spatial_auto_encoder_loss, pixel_reconstruction_error

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


def plot_anomaly(model_name, original_image, reconstructed_image, id, anomaly_type, detection_vector):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """
    save_path = f"anomaly/figure_reconstruction/{anomaly_type}/{model_name}_anomaly_detected_{id}"
    residual_images = original_image - reconstructed_image

    fig_width = original_image.shape[0] * 10
    fig_height = 50  # Adjust as needed
    f, axarr = plt.subplots(3, 10, figsize=(fig_width, fig_height))
    for i in range(original_image.shape[0]):
        axarr[0, i].imshow(original_image[i, 0 , :, :], cmap="gray")
        axarr[1, i].imshow(reconstructed_image[i, 0, :, :], cmap="gray")
        axarr[2, i].imshow(residual_images[i, 0, :, :], cmap="gray")
        axarr[0, i].set_title(f"AD = {detection_vector[i]}", fontsize=64)
    f.suptitle(f'Individual id = {id}, (AD = True => Anomaly detected, else False)', fontsize=80)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path+".pdf")
    # plt.show()
    return 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="darker_circle")
    parser.add_argument("-l", "--loss_input", type=str, required=False, default="pixel")

    args = parser.parse_args()

    anomaly = args.anomaly
    anomaly_list = ["darker_circle", "darker_line", "growing_circle"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'darker_circle', 'darker_line', 'growing_circle' ")
        exit()

    loss_input = args.loss_input
    if loss_input not in ["image", "pixel"]:
        print("Error, anomaly not found, select one of the following anomaly : 'image', 'pixel' ")
        exit()
    if loss_input == "image":
        loss_function = spatial_auto_encoder_loss
    else:
        loss_function = pixel_reconstruction_error

    latent_dimension = 4
    num_workers = round(os.cpu_count()/6)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading anomaly dataset
    anomaly_dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset.csv"
    dataset = Dataset2D(anomaly_dataset_path)
    data_loader = DataLoader(dataset, batch_size=10, num_workers=num_workers, pin_memory=True, shuffle=False)




    ######## TEST WITH VAE ########
    print("Start anomaly detection with VAE")

    # Loading VAE model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model_path = "saved_models_2D/CVAE2D_4_5_100_200.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)


    # Loading thresholds
    threshold_path = f"data_csv/anomaly_threshold_{loss_input}.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)
    VAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    VAE_threshold_99 = threshold_dict["VAE_threshold_99"]


    # These two variables will count the total number of anomalous images detected 
    VAE_anomaly_detected_95 = 0
    VAE_anomaly_detected_99 = 0
    id = 0
    num_images = 0
    with torch.no_grad():
        reconstruction_loss = torch.zeros(10)
        for images in data_loader:
            images = images.to(device)
            mu, logvar, recon_images, _ = model(images)
            for i in range(10):
                num_images += 1
                reconstruction_loss[i], _ = loss_function(mu[i], logvar[i], recon_images[i], images[i])

            detect_anomaly = reconstruction_loss > VAE_threshold_95
            VAE_anomaly_detected_95 += torch.sum(reconstruction_loss > VAE_threshold_95).item()
            VAE_anomaly_detected_99 += torch.sum(reconstruction_loss > VAE_threshold_99).item()
            # print("VAE Reconstruction loss =", reconstruction_loss)
            # print("With threshold_95 :", reconstruction_loss > VAE_threshold_95)
            # print("With threshold_99 :", reconstruction_loss > VAE_threshold_99)
            # print()

            # if (detect_anomaly).any() == True:
            plot_anomaly("VAE", images, recon_images, id, anomaly, detect_anomaly)
            id += 1




    ######## TEST WITH LVAE ########
    print("Start anomaly detection with LVAE \n")

    # Loading anomaly dataset
    transformations = transforms.Compose([])
    dataset = LongitudinalDataset2D(anomaly_dataset_path, read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D, shuffle=False)

    # Loading LVAE model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model_path = "saved_models_2D/CVAE2D_4_5_100_200.pth2"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.training = False

    longitudinal_saving_path = "saved_models_2D/longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path)

    # Loading LVAE thresholds 
    LVAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    LVAE_threshold_99 = threshold_dict["VAE_threshold_99"]

    # These two variables will count the total number of anomalous images detected 
    LVAE_anomaly_detected_95 = 0
    LVAE_anomaly_detected_99 = 0

    # These two variables will be used to know which images isn't classified as anomalous
    LVAE_anomaly_error_95 = torch.zeros((10,10))
    LVAE_anomaly_error_99 = torch.zeros((10,10))
    id = 0
    # TODO: How should we detect anomaly with longitudinal data ?
    with torch.no_grad():
        j = -1
        for data in data_loader:
            j += 1
            images = data[0]
            mus, logvars, recon_images = get_longitudinal_images(data, model, longitudinal_estimator)
            detect_anomaly = torch.zeros(10, dtype=bool)
            for i in range(len(mus)):
                reconstruction_loss, kl_loss = loss_function(mus[i], logvars[i], recon_images[i], images[i])
                detect_anomaly[i] = reconstruction_loss > LVAE_threshold_95

                sum_loss_95 = reconstruction_loss > LVAE_threshold_95
                LVAE_anomaly_error_95[j,i] += not(sum_loss_95)
                sum_loss_99 = reconstruction_loss > LVAE_threshold_99
                LVAE_anomaly_error_99[j,i] += not(sum_loss_99)

                LVAE_anomaly_detected_95 += torch.sum(sum_loss_95).item()
                LVAE_anomaly_detected_99 += torch.sum(sum_loss_99).item()
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

            # if (detect_anomaly).any() == True:
            plot_anomaly("LVAE", images, recon_images, id, anomaly, detect_anomaly)
            id += 1





    ######## PRINTING SOME RESULTS ########

    print()
    print(f"With VAE and {num_images} images:")
    print(f"With threshold_95 we detect {VAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {VAE_anomaly_detected_99} anomaly.")
    print()

    print(f"With LVAE and {num_images} images:")
    print(f"With threshold_95 we detect {LVAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {LVAE_anomaly_detected_99} anomaly.")
    print("With threshold_95, the errors are obtained here :")
    print(LVAE_anomaly_error_95)
    print("With threshold_99, the errors are obtained here :")
    print(LVAE_anomaly_error_95)
    print()

