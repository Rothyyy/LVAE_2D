import numpy as np
import pandas as pd
import torch
from leaspy import AlgorithmSettings, Leaspy
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import os
import json
import matplotlib.pyplot as plt

from dataset.Dataset2D import Dataset2D
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D
from dataset.group_based_train_test_split import group_based_train_test_split

from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

from utils.display_individual_observations_2D import project_encodings_for_results, get_longitudinal_images
from utils.loading_image import open_npy




def plot_comparison(original_image, reconstructed_image_VAE, reconstructed_image_LVAE, id, anomaly_type):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """
    save_path = f"plots/comparison_fig/{anomaly_type}_model_comparison/compare_subject_{id}"
    os.makedirs(f"plots/comparison_fig/{anomaly_type}_model_comparison", exist_ok=True)
    # Compute the residual and binary mask

    mask_threshold = 0.15

    # residual_images_VAE_LVAE = torch.abs(reconstructed_image_VAE - reconstructed_image_LVAE)
    residual_input_VAE = torch.abs(original_image - reconstructed_image_VAE)
    residual_input_LVAE = torch.abs(original_image - reconstructed_image_LVAE)

    # binary_mask = (residual_images_VAE_LVAE > mask_threshold).to(torch.uint8)
    binary_input_VAE = (residual_input_VAE > mask_threshold).to(torch.uint8)
    binary_input_LVAE = (residual_input_LVAE > mask_threshold).to(torch.uint8)
    binary_overlay = torch.zeros((10,64,64,3))
    for i in range(10):
        binary_overlay[i,:,:,0] = binary_input_LVAE[i,:,:]
        binary_overlay[i,:,:,2] = binary_input_VAE[i,:,:]


    fig_width = original_image.shape[0] * 10
    fig_height = 50  # Adjust as needed
    fig, axarr = plt.subplots(4, 10, figsize=(fig_width, fig_height), sharex=True)
    for i in range(original_image.shape[0]):
        axarr[0, i].imshow(original_image[i, 0 , :, :], cmap="gray")
        axarr[0, i].axis('off')
        
        axarr[1, i].imshow(reconstructed_image_VAE[i, 0, :, :], cmap="gray")
        axarr[1, i].axis('off')
        
        axarr[2, i].imshow(reconstructed_image_LVAE[i, 0, :, :], cmap="gray")
        axarr[2, i].axis('off')
        
        # axarr[3, i].imshow(binary_mask[i, 0, :, :], cmap="gray")
        axarr[3, i].imshow(binary_overlay[i])
        # axarr[3, i].imshow(binary_input_VAE[i, 0, :, :], cmap=blue_cmap)
        # axarr[3, i].imshow(binary_input_LVAE[i, 0, :, :], cmap=red_cmap)
        axarr[3, i].axis('off')

    # Row labels
    row_labels = ["Input", "VAE", "LVAE", "Residual \n input - model"]
    for row in range(4):
        # Add label to the first column of each row, closer and vertically centered
        axarr[row, 0].annotate(row_labels[row],
                            xy=(-0.1, 0.5),  # Slightly to the left, centered vertically
                            xycoords='axes fraction',
                            ha='right',
                            va='center',
                            fontsize=60)

    fig.suptitle(f"Comparison VAE/LVAE, Individual id = {id}, anomaly type = {anomaly_type}", fontsize=80)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path+".pdf")
    plt.close(fig)
    return 



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="original")
    parser.add_argument("--dim", type=int, required=False, default=4)
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument("-n", type=int, required=False, default=3)
    args = parser.parse_args()

    anomaly = args.anomaly
    anomaly_list = ["original", "darker_circle", "darker_line", "growing_circle", "shrinking_circle"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'original' 'darker_circle', 'darker_line', 'growing_circle' , 'shrinking_circle' ")
        exit()

    n_samples = args.n

    # Setting some parameters
    beta = args.beta
    latent_dimension = args.dim
    num_workers = round(os.cpu_count()/6)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to dataset and treshold
    if anomaly == "original":
        dataset_path = "data_csv/starmen_train_set.csv"
    else:
        dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset.csv"


    ######## Loading the models ########
    model_VAE_path = f"saved_models_2D/best_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    model_LVAE_path = f"saved_models_2D/best_fold_CVAE2D_{latent_dimension}_{beta}_100_200.pth2"
    longitudinal_saving_path = f"saved_models_2D/best_fold_longitudinal_estimator_params_CVAE2D_{latent_dimension}_{beta}_100_200.json2"

    # Loading VAE model
    model_VAE = CVAE2D_ORIGINAL(latent_dimension)
    model_VAE.load_state_dict(torch.load(model_VAE_path, map_location='cpu'))
    model_VAE = model_VAE.to(device)


    # Loading LVAE model
    model_LVAE = CVAE2D_ORIGINAL(latent_dimension)
    model_LVAE.load_state_dict(torch.load(model_LVAE_path, map_location='cpu'))
    model = model_LVAE.to(device)
    model_LVAE.training = False

    
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path)

    # Loading anomaly dataset and thresholds
    transformations = transforms.Compose([])

    # Loading thresholds and dataset
    # dataset = Dataset2D(anomaly_dataset_path)
    dataset = LongitudinalDataset2D(dataset_path, read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D, shuffle=True)
    
    iteration = 0
    with torch.no_grad():
        
        for data in data_loader:    # Recall: data[0] = images, data[1]= timestamp, data[2] = subject_id

            images = data[0].to(device)
            subject_id = data[2][0]

            mu_VAE, logvar_VAE, recon_images_VAE, _ = model_VAE(images)    # mu.shape = (10,4)
            mus_LVAE, logvars_LVAE, recon_images_LVAE = get_longitudinal_images(data, model_LVAE, longitudinal_estimator)


            plot_comparison(images, recon_images_VAE, recon_images_LVAE, subject_id, anomaly)
            iteration += 1
            if iteration >= n_samples:
                break

