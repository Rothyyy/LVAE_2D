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
    # return encodings, logvars, projected_images
    return torch.from_numpy(predicted_latent_variables[str(subject_id)]), logvars, projected_images



def plot_anomaly_bar(array_anomaly_detected, anomaly_type, method, num_images):
    """
    This function will plot bars corresponding to the number of time the model detect
    an anomaly for the i-th image of a subject.
    """
    save_path = f"anomaly/figure_reconstruction/bar_plots/dataset_{args.dataset}/{anomaly_type}/{freeze_path}/{method}_{anomaly_type}_trajectory_bar_plot.pdf"
    os.makedirs(f"anomaly/figure_reconstruction/bar_plots/dataset_{args.dataset}/{anomaly_type}/{freeze_path}", exist_ok=True)
    x = np.array([i for i in range(1, 10)])

    fig, ax = plt.subplots()
    ax.bar(x, array_anomaly_detected, edgecolor='black')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Count')
    ax.set_title(f'Anomaly detected in timestamp ({int(num_images/10)} images per timestamp)')
    ax.set_xticks(x)
    ax.set_ylim(0, int(num_images/10)+1)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close(fig)
    return 



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="growing_circle")
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument('-f', '--freeze', type=str, required=False, default='y',
                        help='freeze convolution layer ? default = y')
    parser.add_argument('--dataset', type=str, required=True, default="noacc",
                        help='Use the models trained on dataset "acc" or "noacc"')
    parser.add_argument("-kf", type=str, required=False, default="y")
    parser.add_argument("-age", type=str, required=False, default="y")
    args = parser.parse_args()
    if args.freeze == "y":
        freeze_path = "freeze_conv"
    elif args.freeze == "yy":
        freeze_path = "freeze_all"
    else:
        freeze_path = "no_freeze"

    anomaly = args.anomaly
    anomaly_list = ["darker_circle", "darker_line", "growing_circle", "shrinking_circle", "original"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'darker_circle', 'darker_line', 'growing_circle' , 'shrinking_circle' ")
        exit()

    consider_age = "ages" if args.age=="y" else "no_ages"

    # Setting some parameters
    beta = args.beta
    latent_dimension = 4
    num_workers = round(os.cpu_count()/4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to dataset and treshold
    if anomaly == "original":
        anomaly_dataset_path = f"data_csv/starmen_test_set.csv"
    else:
        anomaly_dataset_path = f"data_csv/anomaly_{anomaly}_starmen_dataset.csv"
    threshold_path = f"data_csv/trajectory_threshold_{consider_age}_{freeze_path}_{args.dataset}.json"
    with open(threshold_path) as json_file:
        threshold_dict = json.load(json_file)

    ######## TEST WITH VAE ########
    print("Start anomaly detection")

    # Getting the model's path
    if args.kf == "y" or args.kf == "yy": 
        VAE_model_path = f"saved_models_2D/dataset_{args.dataset}/best_fold_CVAE2D_4_{beta}_100_200.pth"
        LVAE_model_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/best_{freeze_path}_fold_CVAE2D_4_{beta}_100_200.pth2"
        longitudinal_saving_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/best_{freeze_path}_fold_longitudinal_estimator_params_CVAE2D_4_{beta}_100_200.json2"
    else: 
        LVAE_model_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/CVAE2D_4_{beta}_100_200.pth2"
        longitudinal_saving_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/longitudinal_estimator_params_CVAE2D_4_{beta}_100_200.json2"
    

    # Loading LVAE model
    model_LVAE = CVAE2D_ORIGINAL(latent_dimension)
    model_LVAE.load_state_dict(torch.load(LVAE_model_path, map_location='cpu'))
    model_LVAE = model_LVAE.to(device)
    model_LVAE.eval()
    model_LVAE.training = False
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path)

    # Loading anomaly dataset and thresholds
    transformations = transforms.Compose([])
    dataset = LongitudinalDataset2D(anomaly_dataset_path, read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=longitudinal_collate_2D, shuffle=False)
    num_images = len(dataset)*10


    traj_threshold_95 = np.array(threshold_dict["percentile_95_trajectory"])
    traj_threshold_mean = np.array(threshold_dict["mean_trajectory"])
    subject_anomaly = 0

    ages_list = np.array(open("data_starmen/path_to_visit_ages_file.txt", "r").read().split()).astype(float)
    

    with torch.no_grad():
        # This variable will store how many times a image/pixel will be considered as anomalous
        detection_anomaly_LVAE_95 = np.zeros(9) 
        detection_anomaly_LVAE_mean = np.zeros(9) 
        
        for data in data_loader:
            images = data[0]
            images = images.to(device)
            id = data[2][0]
            ages_to_consider_array = ages_list[id*10: id*10+10]
            
            # VAE and LVAE image reconstruction
            mu_VAE, logvar_VAE, _, _ = model_LVAE(images)
            mu_VAE = mu_VAE.numpy()

            # Compute difference between consecutive points
            diffs = mu_VAE[:-1, :] - mu_VAE[1:, :]
            ages_diff = ages_to_consider_array[1:] - ages_to_consider_array[:-1]
            # Compute L1 norm along the last axis
            if consider_age=="ages":
                trajectory_array = np.linalg.norm(diffs, ord=1, axis=1) * ages_diff
            else:
                trajectory_array = np.linalg.norm(diffs, ord=1, axis=1)
            detection_anomaly_LVAE_95 += trajectory_array > traj_threshold_95  
            detection_anomaly_LVAE_mean += trajectory_array > traj_threshold_mean  

            subject_anomaly = subject_anomaly + 1 if (trajectory_array > traj_threshold_95).any() else subject_anomaly

    plot_anomaly_bar(detection_anomaly_LVAE_95, anomaly, "95_percentile", num_images)
    plot_anomaly_bar(detection_anomaly_LVAE_mean, anomaly, "mean", num_images)


    ######## PRINTING SOME RESULTS ########

    print(f"Number of anomaly detected with LVAE and {num_images} images:")
    print(f"With threshold_95 we detect {int(np.sum(detection_anomaly_LVAE_95))} anomalies.")
    print(f"With threshold_mean we detect {int(np.sum(detection_anomaly_LVAE_mean))} anomalies.")
    print(f"The number of anomalous subject is {subject_anomaly}.")


