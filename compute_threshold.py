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

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

from utils_display.display_individual_observations_2D import project_encodings_for_results
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D



transformations = transforms.Compose([])


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


def compute_stats(all_losses, model, method):
    # If necessary we first transform to numpy array and flatten the list
    
    if type(all_losses) == torch.Tensor:
        all_losses = all_losses.numpy()
    elif type(all_losses) == list:
        all_losses = np.array(all_losses)
    if len(all_losses.shape) > 1 and method != "pixel_all":
        all_losses = all_losses.flatten()
    if method == "pixel_all":
        loss_shape = all_losses.shape
        all_losses = all_losses.reshape(loss_shape[0], 64, 64)  # The starmen images have shape 64x64

    # Convert to np.float64 to save stats in json file
    all_losses = all_losses.astype(np.float64)

    stats_dict = {}
    if method != "pixel_all":
        stats_dict[f"{model}_threshold_95"] = np.percentile(all_losses, 95)
        stats_dict[f"{model}_threshold_99"] = np.percentile(all_losses, 99)
        stats_dict[f"{model}_median"] = np.median(all_losses)
        stats_dict[f"{model}_min"] = np.min(all_losses)
        stats_dict[f"{model}_max"] = np.max(all_losses)
        stats_dict[f"{model}_mean"] = np.mean(all_losses)

    else: 
        stats_dict[f"{model}_threshold_95"] = np.percentile(all_losses, 95, axis=0).tolist()
        stats_dict[f"{model}_threshold_99"] = np.percentile(all_losses, 99, axis=0).tolist()
        stats_dict[f"{model}_median"] = np.median(all_losses, axis=0).tolist()
        stats_dict[f"{model}_min"] = np.min(all_losses, axis=0).tolist()
        stats_dict[f"{model}_max"] = np.max(all_losses, axis=0).tolist()
        stats_dict[f"{model}_mean"] = np.mean(all_losses, axis=0).tolist()
    return stats_dict


def plot_recon_error_histogram(recon_error_list, model_name, method):
    save_path = f"anomaly/figure_reconstruction/recon_error/{model_name}_{freeze_path}_{method}_{data_choice}"
    color = "tab:blue" if model_name=="VAE" else "tab:orange"

    if len(recon_error_list.shape) > 1:
        recon_error_list = recon_error_list.flatten() 

    if method == "pixel":
        recon_error_list *= 255

    # Create custom bin labels
    if method == "image":
        custom_bins = [i*15 for i in range(20)]
    else:
        custom_bins = [i*5 for i in range(30)]

    fig, ax = plt.subplots()
    counts, bin_edges, patches = ax.hist(recon_error_list, color=color, edgecolor='black', bins=custom_bins)

    # Set ticks
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges) - 1)]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=45)

    # Add counts on top of each bar
    if method == "image":
        for count, x in zip(counts, bin_centers):
            ax.text(x, count + 0.008 * max(counts), str(int(count)), ha='center', va='bottom', fontsize=10)

    # else:
    #     for count, x in zip(counts, bin_centers):
    #         ax.text(x, count + 0.008 * max(counts), '{:.2e}'.format(count), ha='center', va='bottom', fontsize=10)

    # Add axis labels and title
    ax.set_xlabel('Reconstruction error range')
    ax.set_ylabel('Count')
    if method == "image":
        ax.set_title(f'Reconstruction errors when considering {method} with {model_name} ({data_choice})')
        if data_choice == "train":
            ax.set_ylim(0, 5000)  # Set the y-axis range to fit your expected scale
        else:
            ax.set_ylim(0, 1500)

    elif method == "pixel":
        ax.set_title(f'Pixel differences with {model_name} ({data_choice})')
        if data_choice == "train":
            ax.set_ylim(0, 2.5e7)
        else:
            ax.set_ylim(0, 1e7)

    # Layout fix
    fig.tight_layout()
    plt.savefig(save_path+".pdf")
    plt.close(fig)

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, required=False, default="pixel")
    parser.add_argument("-d", "--dataset", type=str, required=False, default="test")
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument('-f', '--freeze', type=str, required=False, default='y',
                        help='freeze convolution layer ? default = y')
    parser.add_argument('--dataset', type=str, required=False, default="noacc",
                        help='Use the models trained on dataset "acc" or "noacc"')
    args = parser.parse_args()
    freeze_path = "freeze_conv" if args.freeze == 'y' else "no_freeze"

    method = args.method
    if method == "image":
        loss_function = image_reconstruction_error
    elif method == "pixel" or method == "pixel_all":
        loss_function = pixel_reconstruction_error
    else:
        print("Error in the input_loss, select one among the following : ['image', 'pixel', 'pixel_all]")
        exit()
    beta = args.beta
    data_choice = args.dataset
    stats_dict = {}

    # Setting some parameters
    latent_dimension = 4
    num_worker = round(os.cpu_count()/6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nn_saving_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/CVAE2D_4_{beta}_100_200.pth"
    longitudinal_saving_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/longitudinal_estimator_params_CVAE2D_4_{beta}_100_200.json"

    ##### LAUNCHING COMPUTATION FOR VAE #####

    # Loading the VAE mode
    model = CVAE2D_ORIGINAL(latent_dimension)
    model.load_state_dict(torch.load(nn_saving_path, map_location='cpu'))
    model.to(device)
    model.training = False

    if data_choice == "train":
        dataset = Dataset2D("data_csv/starmen_train_set.csv", read_image=open_npy,transform=transformations)
    else:
        dataset = Dataset2D("data_csv/starmen_test_set.csv", read_image=open_npy,transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_worker, shuffle=True, pin_memory=True, )
    all_losses = []

    # 1 epoch to get all reconstruction error with VAE
    with torch.no_grad():
        model.eval()
        for x in data_loader:
            x = x.to(device)

            mu, logvar, recon_x, _ = model(x)
            reconstruction_loss = loss_function(recon_x, x, method)

            loss = reconstruction_loss
            all_losses.append(loss)

    stats_dict.update(compute_stats(all_losses, "VAE", method))
    if method != "pixel_all":
        plot_recon_error_histogram(np.array(all_losses), "VAE", method)



    ##### LAUNCHING COMPUTATION FOR LVAE #####

    # Loading the longitudinal model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model.load_state_dict(torch.load(nn_saving_path + "2", map_location='cpu'))
    model.to(device)
    model.training = False
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path + "2")

    if data_choice == "train":
        dataset = LongitudinalDataset2D("data_csv/starmen_train_set.csv", read_image=open_npy, transform=transformations)
    else:
        dataset = LongitudinalDataset2D("data_csv/starmen_test_set.csv", read_image=open_npy, transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_worker, shuffle=False, collate_fn=longitudinal_collate_2D)
    all_losses = []

    # 1 epoch to get all reconstruction error with LVAE
    with torch.no_grad():
        for data in data_loader:
            x = data[0]
            mus, logvars, recon_x = get_longitudinal_images(data, model, longitudinal_estimator)
            for i in range(len(mus)):
                reconstruction_loss = loss_function(recon_x[i], x[i], method)
                if method == "pixel":
                    reconstruction_loss = reconstruction_loss.flatten()

                all_losses.append(reconstruction_loss)

    stats_dict.update(compute_stats(all_losses, "LVAE", method))
    if method != "pixel_all":
        plot_recon_error_histogram(np.array(all_losses), "LVAE", method)


    # Printing some stats
    if method != "pixel_all":
        print()
        print("Stats for VAE losses :")
        print("min =", stats_dict["VAE_min"])
        print("max =", stats_dict["VAE_max"])
        print("mean =", stats_dict["VAE_mean"])
        print("median =", stats_dict["VAE_median"])
        print("95th percentile =", stats_dict["VAE_threshold_95"])
        print("99th percentile =", stats_dict["VAE_threshold_99"])

        print()
        
        print("Stats for LVAE losses :")
        print("min =", stats_dict["LVAE_min"])
        print("max =", stats_dict["LVAE_max"])
        print("mean =", stats_dict["LVAE_mean"])
        print("median =", stats_dict["LVAE_median"])
        print(f"Number of {method} above VAE_95 =", np.sum(all_losses > stats_dict["VAE_threshold_95"]))
        print("95th percentile =", stats_dict["LVAE_threshold_95"])
        print("99th percentile =", stats_dict["LVAE_threshold_99"])

        print("dict =", stats_dict) 

    # Saving the stats dictionnary in a json file
    if beta == 5:
        with open(f'data_csv/anomaly_threshold_{method}_{freeze_path}.json', 'w') as f:
            json.dump(stats_dict, f, ensure_ascii=False)
    else:
        with open(f'data_csv/anomaly_threshold_{method}_{beta}_{freeze_path}.json', 'w') as f:
            json.dump(stats_dict, f, ensure_ascii=False)
