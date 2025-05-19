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
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

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



def plot_anomaly(model_name, original_image, reconstructed_image, id, anomaly_type, method, detection_vector, pixel_anomaly=None):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """
    save_path = f"anomaly/figure_reconstruction/{anomaly_type}/{method}/{model_name}_subject_{id}"

    # Compute the residual and binary mask
    if method == "image":
        
        residual_images = torch.abs(original_image - reconstructed_image)
        mask_threshold = 0.15
        binary_mask = (residual_images > mask_threshold).to(torch.uint8)

    fig_width = original_image.shape[0] * 10
    fig_height = 50  # Adjust as needed
    f, axarr = plt.subplots(3, 10, figsize=(fig_width, fig_height))
    for i in range(original_image.shape[0]):
        axarr[0, i].imshow(original_image[i, 0 , :, :], cmap="gray")
        axarr[1, i].imshow(reconstructed_image[i, 0, :, :], cmap="gray")

        if method == "image":
            axarr[2, i].imshow(binary_mask[i, 0, :, :], cmap="gray")
        elif method == "pixel":
            axarr[2, i].imshow(pixel_anomaly[i, :].reshape(64,64), cmap="gray")
        else:
            axarr[2, i].imshow(pixel_anomaly[i, :, :], cmap="gray")

        axarr[0, i].set_title(f"AD = {detection_vector[i]}", fontsize=64)
    f.suptitle(f'Individual id = {id}, (AD = True => Anomaly detected, else False)', fontsize=80)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path+".pdf")
    plt.close(f)
    return 


def plot_anomaly_bar(array_anomaly_detected, model_name, anomaly_type, method, num_images):
    """
    This function will plot bars corresponding to the number of time the model detect
    an anomaly for the i-th image of a subject.
    """
    save_path = f"anomaly/figure_reconstruction/bar_plots/{anomaly_type}/{model_name}_{method}_{anomaly_type}_bar_plot"
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

    plt.savefig(save_path+".pdf")
    plt.close(fig)
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
    if loss_input not in ["image", "pixel", "pixel_all"]:
        print("Error, anomaly not found, select one of the following anomaly : 'image', 'pixel', 'pixel_all' ")
        exit()
    if loss_input == "image":
        loss_function = image_reconstruction_error
        size_anomaly = (10, 1)
    else:
        loss_function = pixel_reconstruction_error
        size_anomaly = (10, 64*64) if loss_input == "pixel" else (10, 64, 64)


    # Setting some parameters
    n = 10  # The number of subject to consider
    num_images = n*10
    latent_dimension = 4
    num_workers = round(os.cpu_count()/6)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to dataset and treshold
    anomaly_dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset.csv"
    threshold_path = f"data_csv/anomaly_threshold_{loss_input}.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)



    ######## TEST WITH VAE ########
    print("Start anomaly detection with VAE")

    # Loading VAE model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model_path = "saved_models_2D/CVAE2D_4_5_100_200.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)


    # Loading thresholds and dataset
    dataset = Dataset2D(anomaly_dataset_path)
    data_loader = DataLoader(dataset, batch_size=10, num_workers=num_workers, pin_memory=True, shuffle=False)
    
    VAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    VAE_threshold_99 = threshold_dict["VAE_threshold_99"]

    if loss_input == "pixel_all":
        VAE_threshold_95 = torch.tensor(VAE_threshold_95)
        VAE_threshold_99 = torch.tensor(VAE_threshold_99)


    # These two variables will count the total number of anomalous images/pixel detected 
    VAE_anomaly_detected_95 = 0
    VAE_anomaly_detected_99 = 0
    file_id = 0  # For plot file name


    with torch.no_grad():
        # This variable will store how many times a image/pixel will be considered as anomalous
        total_detection_anomaly = torch.zeros(size_anomaly) 
        
        for images in data_loader:
            pixel_errors = torch.zeros(size_anomaly, dtype=bool) if loss_input != "image" else None

            file_id += 1
            anomaly_detected_vector = torch.zeros(10, dtype=bool) # This vector of boolean will be used for the plot
            images = images.to(device)
            mu, logvar, recon_images, _ = model(images)    # mu.shape = (10,4)

            # For each image of a subject, compute the error and compare with threshold
            for i in range(10):
                reconstruction_error = loss_function(recon_images[i, 0], images[i, 0], loss_input)
                
                compare_to_threshold_95 = reconstruction_error > VAE_threshold_95
                anomaly_detected_vector[i] += (compare_to_threshold_95).any()
                total_detection_anomaly[i] += compare_to_threshold_95

                if loss_input != "image":
                    pixel_errors[i] = compare_to_threshold_95 
                

                VAE_anomaly_detected_95 += torch.sum(reconstruction_error > VAE_threshold_95).item()
                VAE_anomaly_detected_99 += torch.sum(reconstruction_error > VAE_threshold_99).item()

            # For a subject, plot the anomalous image, the reconstructed image and the residual
            plot_anomaly("VAE", images, recon_images, file_id, anomaly, loss_input, anomaly_detected_vector, pixel_errors)

    if loss_input == "image":   # pixel or pixel_all would have too many bar to plot making it unreadable
        plot_anomaly_bar(total_detection_anomaly.flatten(), "VAE", anomaly, loss_input, num_images)






    ######## TEST WITH LVAE ########
    print("Start anomaly detection with LVAE \n")

    # Loading anomaly dataset and thresholds
    transformations = transforms.Compose([])
    dataset = LongitudinalDataset2D(anomaly_dataset_path, read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D, shuffle=False)

    LVAE_threshold_95 = threshold_dict["VAE_threshold_95"]
    LVAE_threshold_99 = threshold_dict["VAE_threshold_99"]
    if loss_input == "pixel_all":
        LVAE_threshold_95 = torch.tensor(LVAE_threshold_95)
        LVAE_threshold_99 = torch.tensor(LVAE_threshold_99)

    # Loading LVAE model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model_path = "saved_models_2D/CVAE2D_4_5_100_200.pth2"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.training = False

    longitudinal_saving_path = "saved_models_2D/longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path)


    # These two variables will count the total number of anomalous images detected 
    LVAE_anomaly_detected_95 = 0
    LVAE_anomaly_detected_99 = 0
    file_id = 0  # For plot file name


    with torch.no_grad():
        # This variable will store how many times a image/pixel will be considered as anomalous
        total_detection_anomaly = torch.zeros(size_anomaly) 

        for data in data_loader:
            pixel_errors = torch.zeros(size_anomaly, dtype=bool) if loss_input != "image" else None

            file_id += 1
            anomaly_detected_vector = torch.zeros(10, dtype=bool) # This vector of boolean will be used for the plot

            images = data[0]
            mus, logvars, recon_images = get_longitudinal_images(data, model, longitudinal_estimator)

            # For each image of a subject, compute the error and compare with threshold
            for i in range(10):
                reconstruction_error = loss_function(recon_images[i, 0], images[i, 0], loss_input)
                
                compare_to_threshold_95 = reconstruction_error > VAE_threshold_95
                anomaly_detected_vector[i] += (compare_to_threshold_95).any()
                total_detection_anomaly[i] += compare_to_threshold_95
                if loss_input != "image":
                    pixel_errors[i] = compare_to_threshold_95 
                

                LVAE_anomaly_detected_95 += torch.sum(reconstruction_error > LVAE_threshold_95).item()
                LVAE_anomaly_detected_99 += torch.sum(reconstruction_error > LVAE_anomaly_detected_99).item()
            
            # For a subject, plot the anomalous image, the reconstructed image and the residual
            plot_anomaly("LVAE", images, recon_images, file_id, anomaly, loss_input, anomaly_detected_vector, pixel_errors)

    if loss_input == "image":   # pixel or pixel_all would have too many bar to plot
        plot_anomaly_bar(total_detection_anomaly.flatten(), "LVAE", anomaly, loss_input, num_images)


    ######## PRINTING SOME RESULTS ########

    print()
    print(f"Using method {loss_input} with VAE and {num_images} images:")
    print(f"With threshold_95 we detect {VAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {VAE_anomaly_detected_99} anomaly.")
    print()

    print(f"Using method {loss_input} with LVAE and {num_images} images:")
    print(f"With threshold_95 we detect {LVAE_anomaly_detected_95} anomaly.")
    print(f"With threshold_99 we detect {LVAE_anomaly_detected_99} anomaly.")


