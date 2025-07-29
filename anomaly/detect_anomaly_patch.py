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

from nnModels.CVAE2D_PATCH import CVAE2D_PATCH
from nnModels.losses import image_reconstruction_error_patch, pixel_reconstruction_error

from utils.display_individual_observations_2D import project_encodings_for_results
from utils.loading_image import open_npy
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch
from .plot_anomaly import plot_anomaly_bar, plot_anomaly_figure_patch
from utils.patch_to_image import pad_array, pixel_counting, patch_to_image



def compute_pixel_ano_score(patch_loss):

    # This array is used to store the anomaly score of one pixel (anomaly score to choose)
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
    # if method == "image":
    #     size_anomaly = (10, 1)


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

    

    ######## TEST WITH VAE ########
    print(f"Start anomaly detection : anomaly={anomaly}, latent dimension={latent_dimension}")

    # Getting the model's path
    model_VAE_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    model_LVAE_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}_{gamma}_{iterations}.pth2"
    longitudinal_saving_path = f"saved_models_2D/best_patch_fold_longitudinal_estimator_params_CVAE2D_{latent_dimension}_{beta}_{gamma}_{iterations}.json2"
  
    # Loading VAE model
    model_VAE = CVAE2D_PATCH(latent_dimension)
    model_VAE.load_state_dict(torch.load(model_VAE_path, map_location='cpu'))
    model_VAE = model_VAE.to(device)
    model_VAE.eval()
    model_VAE.training = False
    model_VAE.to(device)


    # # Loading LVAE model
    # model_LVAE = CVAE2D_PATCH(latent_dimension)
    # model_LVAE.load_state_dict(torch.load(model_LVAE_path, map_location='cpu'))
    # model_LVAE = model_LVAE.to(device)
    # model_LVAE.eval()
    # model_LVAE.training = False
    # model_LVAE.to(device)
    # longitudinal_estimator = Leaspy.load(longitudinal_saving_path)


    # Loading anomaly dataset and thresholds
    # transformations = transforms.Compose([])
    transformations = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32))
            , transforms.Lambda(lambda x: 2*x - 1)
        ])
    dataset = LongitudinalDataset2D_patch(anomaly_dataset_path, read_image=open_npy, transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D_patch, shuffle=False)
    
    VAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    VAE_threshold_99 = threshold_dict["VAE_threshold_99"]

    # LVAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    # LVAE_threshold_99 = threshold_dict["VAE_threshold_99"]


    # These variables will count the total number of anomalous images/pixel detected 
    VAE_anomaly_detected_95 = 0
    VAE_anomaly_detected_99 = 0
    # LVAE_anomaly_detected_95 = 0
    # LVAE_anomaly_detected_99 = 0

    with torch.no_grad():
        # This variable will store how many times a image/pixel will be considered as anomalous
        # total_detection_anomaly_VAE = torch.zeros(size_anomaly).to(device)
        # total_detection_anomaly_LVAE = torch.zeros(size_anomaly).to(device)
        
        for data in data_loader:

            # With batch size = 1, patches contains the patches of the 10 images of 1 patient
            patches = data[0]   # shape = [25000, 1, 15, 15]
            patches = patches.to(device)
            id = data[2][0]

            # pixel_errors_VAE = torch.zeros(size_anomaly).to(device) 
            # pixel_errors_LVAE = torch.zeros(size_anomaly).to(device)
            
            # VAE and LVAE image reconstruction
            _, _, recon_patches_VAE, _ = model_VAE(patches)   # shape = [25000, 1, 15, 15]
            # _, _, recon_images_LVAE = get_longitudinal_images(data, model_LVAE, longitudinal_estimator)


            # Reshape the patches into shape = [10, 2500, 1, 15, 15]
            patches = patches.reshape((10, 2500, 1, 15, 15))
            recon_patches_VAE = recon_patches_VAE.reshape((10, 2500, 1, 15, 15))

            # TODO: It would be faster to simply load the image from the right folders => Have to write the right paths
            image_array_original = np.zeros((10,64,64))  # Array to store image to plot
            image_array_reconstructed = np.zeros((10,64,64))  # Array to store image to plot
            anomaly_map = np.zeros((10, 64, 64), dtype=bool)

            # For each image of a subject, compute the error and compare with threshold
            for t in range(10):
                # List containing the anomaly score of all patches of an image

                ###### Compute VAE's reconstruction error
                anomaly_score_array = loss_function(recon_patches_VAE[t], patches[t]).numpy().reshape((50,50))  # shape = [50, 50]
                anomaly_score_array = pad_array(anomaly_score_array)    # shape = [64, 64]
                pixel_score = anomaly_score_array
                # pixel_score = compute_pixel_ano_score(anomaly_score_array)      # shape = [64, 64]

                # Post processing before getting the image
                patches = (patches + 1) / 2
                recon_patches_VAE = (recon_patches_VAE + 1) / 2
 

                image_array_original[t] = patch_to_image(patches[t, :, 0].numpy())
                image_array_reconstructed[t] = patch_to_image(recon_patches_VAE[t,: , 0].numpy())

                anomaly_map[t] = pixel_score > VAE_threshold_99   #  !!! Here to change which threshold to use
                # total_detection_anomaly_VAE[t] += np.sum(anomaly_map[t])


                ###### Compute LVAE's reconstruction error
                
                # reconstruction_error = loss_function(recon_images_LVAE[t, 0], images[t, 0], method)
                
                # compare_to_threshold = reconstruction_error > VAE_threshold_99   # Here to change with threshold to use
                # anomaly_detected_vector_LVAE[t] += (compare_to_threshold).any()
                # total_detection_anomaly_LVAE[t] += compare_to_threshold
                # if method != "image":
                #     pixel_errors_LVAE[t] = compare_to_threshold 

                # LVAE_anomaly_detected_95 += torch.sum(reconstruction_error > LVAE_threshold_95).detach().cpu().item()
                # LVAE_anomaly_detected_99 += torch.sum(reconstruction_error > LVAE_threshold_99).detach().cpu().item()

            # For a subject, plot the anomalous image, the reconstructed image and the residual
            plot_anomaly_figure_patch(image_array_original, image_array_reconstructed, anomaly_map, id, anomaly, latent_dimension=latent_dimension)


        # if method == "image":   # pixel or pixel_all would have too many bar to plot making it unreadable
            # plot_anomaly_bar(total_detection_anomaly_VAE.flatten(), "VAE", anomaly, method, num_images)
            # plot_anomaly_bar(total_detection_anomaly_LVAE.flatten(), "LVAE", anomaly, method, num_images)

    # if method == "pixel":
    #     total_detection_anomaly_VAE = torch.sum(total_detection_anomaly_VAE.detach().cpu(), dim=1).tolist()
    #     # total_detection_anomaly_LVAE = torch.sum(total_detection_anomaly_LVAE.detach().cpu(), dim=1).tolist()
    #     anomaly_dict_pixel = {}
    #     anomaly_dict_pixel["VAE_pixel_anomaly_99"] = total_detection_anomaly_VAE
    #     # anomaly_dict_pixel["LVAE_pixel_anomaly_99"] = total_detection_anomaly_LVAE
    #     with open(f'./results_pixel_AD_{anomaly}.json', 'w') as f:
    #         json.dump(anomaly_dict_pixel, f, ensure_ascii=False)


    ######## PRINTING SOME RESULTS ########

    # print()
    # print(f"Using method {method} with VAE and {num_images} images:")
    # print(f"With threshold_95 we detect {VAE_anomaly_detected_95} anomaly.")
    # print(f"With threshold_99 we detect {VAE_anomaly_detected_99} anomaly.")
    # print()

    # print(f"Using method {method} with LVAE and {num_images} images:")
    # print(f"With threshold_95 we detect {LVAE_anomaly_detected_95} anomaly.")
    # print(f"With threshold_99 we detect {LVAE_anomaly_detected_99} anomaly.")


