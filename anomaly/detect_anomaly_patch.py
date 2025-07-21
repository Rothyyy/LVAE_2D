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
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

from utils.display_individual_observations_2D import project_encodings_for_results
from utils.loading_image import open_npy
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch
from plot_anomaly import plot_anomaly_bar, plot_anomaly_figure


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="growing_circle")
    parser.add_argument("--method", type=str, required=False, default="pixel")
    parser.add_argument("-n", type=int, required=False, default=5)
    parser.add_argument("--beta", type=float, required=False, default=5)
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
    if method == "image":
        loss_function = image_reconstruction_error
        size_anomaly = (10, 1)
    else:
        loss_function = pixel_reconstruction_error
        size_anomaly = (10, 64*64) if method == "pixel" else (10, 64, 64)


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
    anomaly_dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset.csv"
    threshold_path = f"data_csv/threshold_json/anomaly_threshold_{method}_{latent_dimension}_{beta}.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)



    ######## TEST WITH VAE ########
    print("Start anomaly detection")

    # Getting the model's path
    model_VAE_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    model_LVAE_path = f"saved_models_2D/dataset_{args.dataset}/best_patch_fold_CVAE2D_{latent_dimension}_{beta}_{gamma}_{iterations}.pth2"
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

    # Loading thresholds and dataset
    # dataset = Dataset2D(anomaly_dataset_path)
    # dataset = Dataset2D_VAE_AD(anomaly_dataset_path)
    # data_loader = DataLoader(dataset, batch_size=10, num_workers=num_workers, pin_memory=True, shuffle=False)

    # Loading anomaly dataset and thresholds
    transformations = transforms.Compose([])
    dataset = LongitudinalDataset2D_patch(anomaly_dataset_path, read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D_patch, shuffle=False)
    
    VAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    VAE_threshold_99 = threshold_dict["VAE_threshold_99"]

    if method == "pixel_all":
        VAE_threshold_95 = torch.tensor(VAE_threshold_95)
        VAE_threshold_99 = torch.tensor(VAE_threshold_99)

    # LVAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    # LVAE_threshold_99 = threshold_dict["VAE_threshold_99"]
    # if method == "pixel_all":
    #     LVAE_threshold_95 = torch.tensor(LVAE_threshold_95)
    #     LVAE_threshold_99 = torch.tensor(LVAE_threshold_99)


    # These variables will count the total number of anomalous images/pixel detected 
    VAE_anomaly_detected_95 = 0
    VAE_anomaly_detected_99 = 0
    # LVAE_anomaly_detected_95 = 0
    # LVAE_anomaly_detected_99 = 0

    with torch.no_grad():
        # This variable will store how many times a image/pixel will be considered as anomalous
        total_detection_anomaly_VAE = torch.zeros(size_anomaly).to(device)
        total_detection_anomaly_LVAE = torch.zeros(size_anomaly).to(device)
        
        for data in data_loader:
            images = data[0]
            images = images.to(device)
            id = data[2][0]

            pixel_errors_VAE = torch.zeros(size_anomaly).to(device) if method != "image" else None
            # pixel_errors_LVAE = torch.zeros(size_anomaly).to(device) if method != "image" else None

            # These vectors of boolean will be used for the plot
            anomaly_detected_vector_VAE = torch.zeros(10, dtype=bool).to(device)   
            anomaly_detected_vector_LVAE = torch.zeros(10, dtype=bool).to(device)  # This vector of boolean will be used for the plot
            
            # VAE and LVAE image reconstruction
            mu_VAE, logvar_VAE, recon_images_VAE, _ = model_VAE(images)    # mu.shape = (10,4)
            # mus_LVAE, logvars_LVAE, recon_images_LVAE = get_longitudinal_images(data, model_LVAE, longitudinal_estimator)

            # For each image of a subject, compute the error and compare with threshold
            for i in range(10):

                # Compute VAE's reconstruction error
                reconstruction_error_VAE = loss_function(recon_images_VAE[i, 0], images[i, 0], method)
                
                compare_to_threshold = reconstruction_error_VAE > VAE_threshold_99   #  !!! Here to change which threshold to use
                anomaly_detected_vector_VAE[i] += (compare_to_threshold).any()
                total_detection_anomaly_VAE[i] += compare_to_threshold

                if method != "image":
                    pixel_errors_VAE[i] = compare_to_threshold 

                VAE_anomaly_detected_95 += torch.sum(reconstruction_error_VAE > VAE_threshold_95).detach().cpu().item()
                VAE_anomaly_detected_99 += torch.sum(reconstruction_error_VAE > VAE_threshold_99).detach().cpu().item()

                # Compute LVAE's reconstruction error
                # reconstruction_error = loss_function(recon_images_LVAE[i, 0], images[i, 0], method)
                
                # compare_to_threshold = reconstruction_error > VAE_threshold_99   # Here to change with threshold to use
                # anomaly_detected_vector_LVAE[i] += (compare_to_threshold).any()
                # total_detection_anomaly_LVAE[i] += compare_to_threshold
                # if method != "image":
                #     pixel_errors_LVAE[i] = compare_to_threshold 

                # LVAE_anomaly_detected_95 += torch.sum(reconstruction_error > LVAE_threshold_95).detach().cpu().item()
                # LVAE_anomaly_detected_99 += torch.sum(reconstruction_error > LVAE_threshold_99).detach().cpu().item()

            # For a subject, plot the anomalous image, the reconstructed image and the residual
            # plot_anomaly(images, recon_images_VAE, recon_images_LVAE,
            #               id, anomaly, method,
            #               anomaly_detected_vector_VAE, anomaly_detected_vector_LVAE, 
            #               pixel_errors_VAE, pixel_errors_LVAE)

    if method == "image":   # pixel or pixel_all would have too many bar to plot making it unreadable
        plot_anomaly_bar(total_detection_anomaly_VAE.flatten(), "VAE", anomaly, method, num_images)
        # plot_anomaly_bar(total_detection_anomaly_LVAE.flatten(), "LVAE", anomaly, method, num_images)

    if method == "pixel":
        total_detection_anomaly_VAE = torch.sum(total_detection_anomaly_VAE.detach().cpu(), dim=1).tolist()
        # total_detection_anomaly_LVAE = torch.sum(total_detection_anomaly_LVAE.detach().cpu(), dim=1).tolist()
        anomaly_dict_pixel = {}
        anomaly_dict_pixel["VAE_pixel_anomaly_99"] = total_detection_anomaly_VAE
        # anomaly_dict_pixel["LVAE_pixel_anomaly_99"] = total_detection_anomaly_LVAE
        with open(f'./results_pixel_AD_{anomaly}.json', 'w') as f:
            json.dump(anomaly_dict_pixel, f, ensure_ascii=False)


    ######## PRINTING SOME RESULTS ########

    print()
    print(f"Using method {method} with VAE and {num_images} images:")
    print(f"With threshold_95 we detect {VAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {VAE_anomaly_detected_99} anomaly.")
    print()

    # print(f"Using method {method} with LVAE and {num_images} images:")
    # print(f"With threshold_95 we detect {LVAE_anomaly_detected_95} anomaly.")
    # print(f"With threshold_99 we detect {LVAE_anomaly_detected_99} anomaly.")


