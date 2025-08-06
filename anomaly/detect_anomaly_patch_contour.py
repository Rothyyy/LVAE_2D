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

from dataset.Dataset2D import Dataset2D_patch
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from dataset.group_based_train_test_split import group_based_train_test_split

from nnModels.CVAE2D_PATCH import CVAE2D_PATCH, CVAE2D_PATCH_16, CVAE2D_PATCH_32, CVAE2D_PATCH_3latent64, CVAE2D_PATCH_3latent32, CVAE2D_PATCH_7
from nnModels.losses import image_reconstruction_error_patch, pixel_reconstruction_error

from utils.display_individual_observations_2D import project_encodings_for_results
from utils.loading_image import open_npy
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch
from dataset.LongitudinalDataset2D_patch_contour import LongitudinalDataset2D_patch_contour, longitudinal_collate_2D_patch_contour
from .plot_anomaly import plot_anomaly_bar, plot_anomaly_figure_patch
from utils.patch_to_image import pad_array, pixel_counting, patch_to_image, patch_contour_to_image

np.set_printoptions(threshold=np.inf)

def compute_pixel_ano_score(patch_loss):

    # This array is used to store the anomaly score of one pixel
    pixel_anomaly_score = np.zeros((64,64))

    # This array was obtained previously after putting the image in the model and getting the loss/anomaly score
    if patch_loss.shape != (64,64):
        patch_loss = np.array(patch_loss).reshape((50,50))
        patch_loss = pad_array(patch_loss)


    for i in range(64):
        for j in range(64):
            # Build the window to get all the anomaly patches
            top = max(i - 15//2, 0)
            bottom = min(i + 15//2 + 1, 64)
            left = max(j - 15//2, 0)
            right = min(j + 15//2 + 1, 64)

            pixel_anomaly_score[i,j] += np.sum(patch_loss[top:bottom , left:right])     # Just compute sum
            pixel_anomaly_score[i,j] += np.sum(patch_loss[top:bottom , left:right])/pixel_counting[i,j]     # With mean
             
    return pixel_anomaly_score


def compute_anomaly_image(anomaly_scores, centers):
    """
    From the 1D anomaly_scores build the 64x64 array where the 
    coordinate (x,y) contains the corresponding patch anomaly score.
    """

    anomaly_map = np.zeros((64,64))
    patch_processed = 0
    for x,y in centers:
        anomaly_map[x, y] = anomaly_scores[patch_processed]
        patch_processed += 1

    return anomaly_map

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="growing_circle")
    parser.add_argument("--method", type=str, required=False, default="image")
    parser.add_argument("-n", type=int, required=False, default=5)
    parser.add_argument("--beta", type=float, required=False, default=2)
    parser.add_argument("--gamma", type=float, required=False, default=100)
    parser.add_argument("--iterations", type=int, required=False, default=5)
    parser.add_argument("--dim", type=int, required=False, default=64)
    args = parser.parse_args()


    anomaly = args.anomaly
    anomaly_list = ["darker_circle", "darker_line", "growing_circle", "shrinking_circle"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'darker_circle', 'darker_line', 'growing_circle' , 'shrinking_circle' ")
        exit()

    method = args.method
    if method not in ["image", "pixel", "pixel_all"]:
        print("Error, anomaly not found, select one of the following anomaly : 'image', 'pixel', 'pixel_all' ")
        exit()
    loss_function = image_reconstruction_error_patch

    # Setting some parameters
    beta = args.beta
    gamma = args.gamma
    iterations = args.iterations

    n = args.n  # The number of subject to consider
    num_images = n*10
    latent_dimension = args.dim
    num_workers = round(os.cpu_count()/6)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to dataset and treshold
    anomaly_dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset_patch.csv"
    threshold_path = f"data_csv/threshold_json/anomaly_threshold_patch_{method}_{latent_dimension}_{beta}.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)

    if latent_dimension == 332:
        model_type = CVAE2D_PATCH_3latent32
    elif latent_dimension == 32:
        model_type = CVAE2D_PATCH_32
    elif latent_dimension == 364:
        model_type = CVAE2D_PATCH_3latent64
    elif latent_dimension == 7:
        model_type = CVAE2D_PATCH_7
    else:
        model_type = CVAE2D_PATCH_16

    ######## TEST WITH VAE ########
    print(f"Start anomaly detection : anomaly={anomaly}, latent dimension={latent_dimension}")

    # Getting the model's path
    model_VAE_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    model_LVAE_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}_{gamma}_{iterations}.pth2"
    longitudinal_saving_path = f"saved_models_2D/best_patch_fold_longitudinal_estimator_params_CVAE2D_{latent_dimension}_{beta}_{gamma}_{iterations}.json2"
  
    # Loading VAE model
    model_VAE = model_type(latent_dimension)
    model_VAE.load_state_dict(torch.load(model_VAE_path, map_location='cpu'))
    model_VAE.eval()
    model_VAE.training = False
    model_VAE = model_VAE.to(device)

    # Loading anomaly dataset and thresholds
    transformations = transforms.Compose([])

    dataset = LongitudinalDataset2D_patch_contour(anomaly_dataset_path, read_image=open_npy, transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D_patch_contour, shuffle=False)

    VAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    VAE_threshold_99 = threshold_dict["VAE_threshold_99"]

    # These variables will count the total number of anomalous images/pixel detected 
    VAE_anomaly_detected_95 = 0
    VAE_anomaly_detected_99 = 0

    with torch.no_grad():
        
        for data in data_loader:

            # With batch size = 1, patches contains the patches of the 10 images of 1 patient
            patches = data[0]   # torch tensor containing the patient patches of all 10 images. Use centers to know which patch belong to what image !
            patches = patches.to(device)
            id = data[2][0] 
            centers = data[3][0]    # List of 10 elements, each element is a numpy array containing the centers coordinate of the image's patches
            sum_patch_center = 0
            for i in range(len(centers)):
                sum_patch_center += len(centers[i])
            
            # VAE and LVAE image reconstruction
            _, _, recon_patches_VAE, _ = model_VAE(patches)     # shape = [num_patches, 1, patch_size, patch_size]
            patch_anomaly_score = loss_function(recon_patches_VAE, patches).numpy()


            # TODO: It would be faster to simply load the image from the right folders => Have to write the right paths
            image_array_original = np.zeros((10,64,64))         # Array to store image to plot
            image_array_reconstructed = np.zeros((10,64,64))    # Array to store image to plot
            anomaly_map = np.zeros((10, 64, 64), dtype=bool)

            patches = (patches.squeeze(1)).numpy()
            recon_patches_VAE = (recon_patches_VAE.squeeze(1)).numpy()
            patch_processed = 0   
            # For each image of a subject, compute the error and compare with threshold
            for t in range(10):

                # Getting the image's centers and anomaly scores
                image_centers = centers[t]
                num_patches = image_centers.shape[0]
                patch_image_t = patches[patch_processed: patch_processed+num_patches]
                recon_patches_VAE_t = recon_patches_VAE[patch_processed: patch_processed+num_patches]
                image_anomaly_score = patch_anomaly_score[patch_processed: patch_processed+num_patches]
                patch_processed += num_patches

                ###### Compute anomaly map
                
                anomaly_score_map = compute_anomaly_image(image_anomaly_score, image_centers)    # shape = [64, 64]
                
                image_array_original[t] = patch_contour_to_image(patch_image_t, image_centers)
                image_array_reconstructed[t] = patch_contour_to_image(recon_patches_VAE_t, image_centers)

                anomaly_map[t] = anomaly_score_map > VAE_threshold_95   #  !!! Here to change which threshold to use
                # anomaly_map[t] = anomaly_score_map > 5   #  !!! Here to change which threshold to use



            # For a subject, plot the anomalous image, the reconstructed image and the residual
            plot_anomaly_figure_patch(image_array_original, image_array_reconstructed, anomaly_map, id, anomaly, latent_dimension=latent_dimension)


