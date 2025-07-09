import numpy as np
import pandas as pd
import sklearn.manifold
import torch
from leaspy import AlgorithmSettings, Leaspy
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import os
import json
import matplotlib.pyplot as plt
import sklearn

from dataset.Dataset2D import Dataset2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from dataset.group_based_train_test_split import group_based_train_test_split

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

from utils_display.display_individual_observations_2D import project_encodings_for_results
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D

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
    # return torch.from_numpy(predicted_latent_variables[str(subject_id)]), logvars, projected_images


def plot_figures(original_image, reconstructed_original_LVAE, anomaly_image, reconstructed_anomaly_LVAE, id, anomaly_type, show_fig=False):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """

    fig_width = original_image.shape[0] * 10
    fig_height = 50  # Adjust as needed
    f, axarr = plt.subplots(4, 10, figsize=(fig_width, fig_height))
    for i in range(original_image.shape[0]):
        axarr[0, i].imshow(original_image[i, :, :], cmap="gray")
        axarr[1, i].imshow(reconstructed_original_LVAE[i, :, :], cmap="gray")
        axarr[2, i].imshow(anomaly_image[i, :, :], cmap="gray")
        axarr[3, i].imshow(reconstructed_anomaly_LVAE[i, :, :], cmap="gray")

        # axarr[0, i].set_title(f"LVAE={detection_vector_LVAE[i]}", fontsize=50)
    
    # Row labels
    row_labels = ["Original", "Original\n reconstruction", "Anomaly", "Anomaly\n reconstruction"]
    for row in range(4):
        # Add label to the first column of each row, closer and vertically centered
        axarr[row, 0].annotate(row_labels[row],
                            xy=(-0.1, 0.5),  # Slightly to the left, centered vertically
                            xycoords='axes fraction',
                            ha='right',
                            va='center',
                            fontsize=60)

    f.suptitle(f'Individual id = {id}', fontsize=80)
    plt.tight_layout()
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"visu/{id}_plot_figures_{anomaly_type}.pdf")
    plt.clf()
    return 

def compute_trajectory_sequence_threshold(encodings_array, ages_array, with_ages=True):
    """
    This functions takes as input the array of all the dataset's encodings and will return
    a dictionnary containing statistics on the trajectory. 

    Input:
        encodings_array: a numpy array of shape (num_subject, num_images, encoding_size), usually (200, 10, 4)
        ages_array: a numpy array containing the timestamp of each images 
    """
    num_subject = encodings_array.shape[0]
    trajectory_array = np.zeros((num_subject, encodings_array.shape[1]-1))

    # Compute difference between consecutive points
    encodings_diffs = encodings_array[:, :-1, :] - encodings_array[:, 1:, :]
    ages_diff = ages_array[: , 1:] - ages_array[:, :-1]

    # Compute L1 norm along the last axis
    if with_ages:
        trajectory_array = np.linalg.norm(encodings_diffs, ord=1, axis=2)*np.abs(ages_diff)
    else:
        trajectory_array = np.linalg.norm(encodings_diffs, ord=1, axis=2)

    traj_dict = {}
    traj_dict["mean_trajectory"] = np.mean(trajectory_array, axis=0).tolist()
    traj_dict["percentile_95_trajectory"] = np.percentile(trajectory_array, 95, axis=0).tolist()
    traj_dict["max_trajectory"] = np.max(trajectory_array, axis=0).tolist()
    traj_dict["min_trajectory"] = np.min(trajectory_array, axis=0).tolist()
    traj_dict["median_trajectory"] = np.median(trajectory_array, axis=0).tolist()

    return traj_dict

def compute_trajectory_threshold(encodings_array, ages_array, with_ages=True):
    """
    This functions takes as input the array of all the dataset's encodings and will return
    a dictionnary containing statistics on the trajectory. 

    Input:
        encodings_array: a numpy array of shape (num_subject, num_images, encoding_size), usually (200, 10, 4)
        ages_array: a numpy array containing the timestamp of each images 
    """
    num_subject = encodings_array.shape[0]
    trajectory_array = np.zeros((num_subject, encodings_array.shape[1]-1))

    # Compute difference between consecutive points
    encodings_diffs = encodings_array[:, :-1, :] - encodings_array[:, 1:, :]
    ages_diff = ages_array[: , 1:] - ages_array[:, :-1]

    # Compute L1 norm along the last axis
    if with_ages:
        trajectory_array = np.linalg.norm(encodings_diffs, ord=1, axis=2)*np.abs(ages_diff)
    else:
        trajectory_array = np.linalg.norm(encodings_diffs, ord=1, axis=2)

    traj_dict = {}
    traj_dict["mean_trajectory"] = np.mean(trajectory_array.flatten())
    traj_dict["percentile_95_trajectory"] = np.percentile(trajectory_array.flatten(), 95)
    traj_dict["max_trajectory"] = np.max(trajectory_array.flatten())
    traj_dict["min_trajectory"] = np.min(trajectory_array.flatten())
    traj_dict["median_trajectory"] = np.median(trajectory_array.flatten())

    return traj_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument("-kf", type=str, required=False, default="y")
    parser.add_argument('-f', '--freeze', type=str, required=True, default='y',
                        help='freeze convolution layer ? default = y')   
    parser.add_argument('--dataset', type=str, required=True, default="noacc",
                        help='Use the models trained on dataset "acc" or "noacc"')
    parser.add_argument('--anomaly', '-a', type=str, required=False, default="growing_circle")
    parser.add_argument('-age', type=str, required=False, default="y")
    args = parser.parse_args()

    dataset_name = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    consider_age = True if args.age=="y" else False

    if args.freeze == "y":
        freeze_path = "freeze_conv"
    elif args.freeze == "yy":
        freeze_path = "freeze_all"
    else:
        freeze_path = "no_freeze"

    if args.kf == "y" or args.kf == "yy":
        VAE_model_path = f"saved_models_2D/dataset_{args.dataset}/best_fold_CVAE2D_4_5_100_200.pth"
        LVAE_model_path = f"saved_models_2D/dataset_{dataset_name}/{freeze_path}/best_{freeze_path}_fold_CVAE2D_4_5_100_200.pth2"
        LVAE_estimator_path = f"saved_models_2D/dataset_{dataset_name}/{freeze_path}/best_{freeze_path}_fold_longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"
    else:
        LVAE_model_path = f"saved_models_2D/dataset_{dataset_name}/{freeze_path}/CVAE2D_4_5_100_200.pth2"
        LVAE_estimator_path = f"saved_models_2D/dataset_{dataset_name}/{freeze_path}/longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"

    data_csv_path = "data_csv/starmen_test_set.csv"
    dataset_test = LongitudinalDataset2D(data_csv_path)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=longitudinal_collate_2D, num_workers=round(os.cpu_count()/4))

    # anomaly_type = args.anomaly
    # data_anomaly_csv_path = f"data_csv/anomaly_{anomaly_type}_starmen_dataset.csv"
    # dataset_anomaly = LongitudinalDataset2D(data_anomaly_csv_path)
    # dataloader_anomaly = DataLoader(dataset_anomaly, batch_size=1, shuffle=False, collate_fn=longitudinal_collate_2D, num_workers=round(os.cpu_count()/4))

    model = CVAE2D_ORIGINAL(4)
    model.gamma = 100
    model.beta = 5
    model.load_state_dict(torch.load(LVAE_model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    model.training = False

    longitudinal_estimator = Leaspy.load(LVAE_estimator_path)

    encodings_original_array = np.zeros((len(dataset_test), 10, 4))
    # encodings_anomaly_array = np.zeros((len(dataset_test), 10, 4))
    n_subject = 0

    image_original_array = np.zeros((len(dataset_test), 10, 64, 64))
    image_original_reconstructed_array = np.zeros((len(dataset_test), 10, 64, 64))

    ages_list = np.array(open("data_starmen/path_to_visit_ages_file.txt", "r").read().split()).astype(float)
    ages_to_consider_array = np.zeros((len(dataset_test), 10))
    # image_anomaly_array = np.zeros((len(dataset_test), 10, 64, 64))
    # image_anomaly_reconstructed_array = np.zeros((len(dataset_test), 10, 64, 64))
    n_subject = 0

    # We do an epoch to get all the encodings and ages
    for data in dataloader_test:
        subject_id = data[2][0]

        ages_to_consider_array[n_subject, :] = ages_list[subject_id*10: subject_id*10+10]

        images_original = data[0].to(device)
        image_original_array[n_subject] = images_original.detach().numpy().reshape((10,64,64))
        mus_original_LVAE, logvars_original_LVAE, reconstructed_original, _ = model(images_original)

        # images_anomaly = dataset_anomaly.get_images_from_id(subject_id)[0].reshape((10,1,64,64)).to(device)
        # image_anomaly_array[n_subject] = images_anomaly.detach().numpy().reshape((10,64,64))        
        # mus_anomaly_LVAE, logvars_original_LVAE, reconstructed_anomaly, _ = model(images_anomaly)
        
        for i in range(10):
            encodings_original_array[n_subject, i, :] = mus_original_LVAE[i].detach().numpy()
            # image_original_reconstructed_array[n_subject, i] = reconstructed_original[i].detach().numpy()

            # encodings_anomaly_array[n_subject, i, :] = mus_anomaly_LVAE[i].detach().numpy()
            # image_anomaly_reconstructed_array[n_subject, i] = reconstructed_anomaly[i].detach().numpy()

        n_subject += 1

    # traj_stats = compute_trajectory_threshold(encodings_original_array, ages_to_consider_array, consider_age)
    traj_stats = compute_trajectory_sequence_threshold(encodings_original_array, ages_to_consider_array, consider_age)
    if consider_age:
        with open(f'data_csv/trajectory_threshold_ages_{freeze_path}_{dataset_name}.json', 'w') as f:
            json.dump(traj_stats, f, ensure_ascii=False)
    else:
        with open(f'data_csv/trajectory_threshold_no_ages_{freeze_path}_{dataset_name}.json', 'w') as f:
            json.dump(traj_stats, f, ensure_ascii=False)




