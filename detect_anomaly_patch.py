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

from utils_display.display_individual_observations_2D import project_encodings_for_results
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch

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


def plot_anomaly(original_image, reconstructed_image_VAE, reconstructed_image_LVAE,
                  id, anomaly_type, method, 
                  detection_vector_VAE, detection_vector_LVAE, pixel_anomaly_VAE=None, pixel_anomaly_LVAE=None):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """
    os.makedirs(f"anomaly/figure_reconstruction/dataset_{args.dataset}/{anomaly_type}/{method}", exist_ok=True)
    save_path = f"anomaly/figure_reconstruction/dataset_{args.dataset}/{anomaly_type}/{method}/AD_subject_{id}.pdf"

    # Compute the residual and binary mask
    if method == "image":
        residual_images_VAE = torch.abs(original_image - reconstructed_image_VAE)
        residual_images_LVAE = torch.abs(original_image - reconstructed_image_LVAE)
        mask_threshold = 0.15
        binary_mask_VAE = (residual_images_VAE > mask_threshold).to(torch.uint8)
        binary_mask_LVAE = (residual_images_LVAE > mask_threshold).to(torch.uint8)
        binary_overlay = torch.zeros((10,64,64,3))
        for i in range(10):
            binary_overlay[i,:,:,0] = binary_mask_LVAE[i,:,:]
            binary_overlay[i,:,:,2] = binary_mask_VAE[i,:,:]

    elif method == "pixel_all":
        binary_mask_LVAE = pixel_anomaly_LVAE.to(torch.uint8)
        binary_mask_VAE = pixel_anomaly_VAE.to(torch.uint8)
        binary_overlay = torch.zeros((10,64,64,3))
        for i in range(10):
            binary_overlay[i,:,:,0] = binary_mask_LVAE[i,:,:]
            binary_overlay[i,:,:,2] = binary_mask_VAE[i,:,:]
    
    else: # method == "pixel"
        binary_mask_LVAE = pixel_anomaly_LVAE.reshape(10,64,64).to(torch.uint8)
        binary_mask_VAE = pixel_anomaly_VAE.reshape(10,64,64).to(torch.uint8)
        binary_overlay = torch.zeros((10,64,64,3))
        for i in range(10):
            binary_overlay[i,:,:,0] = binary_mask_LVAE[i,:,:]
            binary_overlay[i,:,:,2] = binary_mask_VAE[i,:,:]

    fig_width = original_image.shape[0] * 10
    fig_height = 50  # Adjust as needed
    f, axarr = plt.subplots(4, 10, figsize=(fig_width, fig_height))
    for i in range(original_image.shape[0]):
        axarr[0, i].imshow(original_image[i, 0 , :, :], cmap="gray")
        axarr[1, i].imshow(reconstructed_image_VAE[i, 0, :, :], cmap="gray")
        axarr[2, i].imshow(reconstructed_image_LVAE[i, 0, :, :], cmap="gray")
        axarr[3, i].imshow(binary_overlay[i])

        if method=="image":
            axarr[0, i].set_title(f"VAE={detection_vector_VAE[i]}, LVAE={detection_vector_LVAE[i]}", fontsize=50)
        else:
            axarr[0, i].set_title(f"VAE={int(torch.sum(pixel_anomaly_VAE[i]).item())}, LVAE={int(torch.sum(pixel_anomaly_LVAE[i]).item())}", fontsize=50)

    # Row labels
    row_labels = ["Input", "VAE", "LVAE", f"Residual \n input-model"]
    for row in range(4):
        # Add label to the first column of each row, closer and vertically centered
        axarr[row, 0].annotate(row_labels[row],
                            xy=(-0.1, 0.5),  # Slightly to the left, centered vertically
                            xycoords='axes fraction',
                            ha='right',
                            va='center',
                            fontsize=60)

    if method=="image":
        f.suptitle(f'Individual id = {id}, method = {method}, (model = True => Anomaly detected, else False)', fontsize=80)
    else:
        f.suptitle(f'Individual id = {id}, method = {method}, (model = x = # anomalous pixel)', fontsize=80)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(f)
    return 


def plot_anomaly_bar(array_anomaly_detected, model_name, anomaly_type, method, num_images):
    """
    This function will plot bars corresponding to the number of time the model detect
    an anomaly for the i-th image of a subject.
    """
    save_path = f"anomaly/figure_reconstruction/bar_plots/dataset_{args.dataset}/{anomaly_type}/{model_name}_{method}_{anomaly_type}_bar_plot.pdf"
    os.makedirs(f"anomaly/figure_reconstruction/bar_plots/dataset_{args.dataset}/{anomaly_type}", exist_ok=True)
    x = np.array([i for i in range(1, 11)])
    color = "tab:blue" if model_name=="VAE" else "tab:orange"

    fig, ax = plt.subplots()
    ax.bar(x, array_anomaly_detected, color=color, edgecolor='black')
    ax.set_xlabel('Image')
    ax.set_ylabel('Count')
    ax.set_title(f'Anomaly detected in images ({int(num_images/10)} images per timestamp)')
    ax.set_xticks(x)
    ax.set_ylim(0, int(num_images/10)+1)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close(fig)
    return 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="growing_circle")
    parser.add_argument("-m", "--method", type=str, required=False, default="pixel")
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument("--dim", type=int, required=False, default=3)
    parser.add_argument('--dataset', type=str, required=True, default="noacc",
                        help='Use the models trained on dataset "acc" or "noacc"')
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
    n = 200  # The number of subject to consider
    num_images = n*10
    latent_dimension = args.dim
    num_workers = round(os.cpu_count()/6)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to dataset and treshold
    anomaly_dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset.csv"
    if beta != 5:
        threshold_path = f"data_csv/anomaly_threshold_{method}_{beta}_{args.dataset}.json"
    else:
        threshold_path = f"data_csv/anomaly_threshold_{method}_{args.dataset}.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)



    ######## TEST WITH VAE ########
    print("Start anomaly detection")

    # Getting the model's path
    model_VAE_path = f"saved_models_2D/dataset_{args.dataset}/best_patch_fold_CVAE2D_{latent_dimension}_{beta}_100_20.pth"
    model_LVAE_path = f"saved_models_2D/dataset_{args.dataset}/best_patch_fold_CVAE2D_{latent_dimension}_{beta}_100_20.pth2"
    longitudinal_saving_path = f"saved_models_2D/dataset_{args.dataset}/best_patch_fold_longitudinal_estimator_params_CVAE2D_{latent_dimension}_{beta}_100_20.json2"
  
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


