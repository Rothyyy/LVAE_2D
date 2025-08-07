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

from dataset.Dataset2D import Dataset2D_patch, collate_2D_patch
from dataset.LongitudinalDataset2D_patch_contour import Dataset
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch

from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn

from nnModels.CVAE2D_PATCH import CVAE2D_PATCH, CVAE2D_PATCH_16, CVAE2D_PATCH_32, CVAE2D_PATCH_64, CVAE2D_PATCH_4latent64, CVAE2D_PATCH_3latent32, CVAE2D_PATCH_7 
from nnModels.losses import image_reconstruction_error_patch, pixel_reconstruction_error

from utils.display_individual_observations_2D import project_encodings_for_results, get_longitudinal_images
from utils.loading_image import open_npy



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

    return stats_dict


def plot_recon_error_histogram_patch(recon_error_list, model_name, method):
    
    save_path = f"plots/recon_error/hist_patch_{model_name}_{method}_{latent_dimension}.pdf"
    os.makedirs(f"plots/recon_error/", exist_ok=True)
    color = "tab:blue" if model_name=="VAE" else "tab:orange"

    if len(recon_error_list.shape) > 1:
        recon_error_list = recon_error_list.flatten() 

    if method == "pixel":
        recon_error_list *= 255

    # Create custom bin labels
    custom_bins = [i for i in range(20)]

    fig, ax = plt.subplots()
    counts, bin_edges, patches = ax.hist(recon_error_list, color=color, edgecolor='black', bins=custom_bins)
    # Set ticks
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges) - 1)]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=45)

    # Add counts on top of each bar
    # if method == "image":
    #     for count, x in zip(counts, bin_centers):
    #         ax.text(x, count + 0.008 * max(counts), str(int(count)), ha='center', va='bottom', fontsize=10)

    # Add axis labels and title
    ax.set_xlabel('Reconstruction error range')
    ax.set_ylabel('Count')
    # ax.set_title(f'Reconstruction errors when considering patches with {model_name} (dim {latent_dimension})')
    ax.set_ylim(0, 0.5e6)

    # Layout fix
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=False, default="image")
    parser.add_argument("--dim", type=int, required=False, default=64)
    parser.add_argument("--beta", type=float, required=False, default=2)
    parser.add_argument("--gamma", type=float, required=False, default=100)
    parser.add_argument("--iter", type=int, required=False, default=5)
    parser.add_argument("--size", type=int, required=False, default=15)
    parser.add_argument("--plot", type=bool, required=False, default=False)
    args = parser.parse_args()


    method = args.method
    if method == "image":
        loss_function = image_reconstruction_error_patch
    elif method == "pixel" or method == "pixel_all":
        loss_function = pixel_reconstruction_error
    else:
        print("Error in the input_loss, select one among the following : ['image', 'pixel', 'pixel_all]")
        exit()


    stats_dict = {}

    # Setting some parameters
    patch_size = args.size
    latent_dimension = args.dim
    train_iter = args.iter
    beta = args.beta
    gamma = args.gamma
    num_worker = round(os.cpu_count()/6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If true then we plot from a saved list of losses
    if args.plot:
        print("Plot from previous all_losses file only !")
        with open(f'data_csv/VAE_patch_losses_{latent_dimension}.json', 'r') as f:
            all_losses = json.load(f)
        plot_recon_error_histogram_patch(np.array(all_losses), "VAE", method)
        exit()

    if latent_dimension == 332:
        model_type = CVAE2D_PATCH_3latent32
    elif latent_dimension == 32:
        model_type = CVAE2D_PATCH_32
    elif latent_dimension == 464:
        model_type = CVAE2D_PATCH_4latent64
    elif latent_dimension == 64:
        model_type = CVAE2D_PATCH_64
    elif latent_dimension == 7:
        model_type = CVAE2D_PATCH_7
    else:
        model_type = CVAE2D_PATCH_16

    # Getting the path to the saved models
    VAE_nn_saving_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    LVAE_nn_saving_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}_{gamma}_{train_iter}.pth2"
    longitudinal_saving_path = f"saved_models_2D/best_patch_fold_longitudinal_estimator_params_CVAE2D_{latent_dimension}_{beta}_{gamma}_{train_iter}.json2"


    transformations = transforms.Compose([])
    # transformations = transforms.Compose([
    #         transforms.Lambda(lambda x: x.to(torch.float32))
    #         , transforms.Lambda(lambda x: 2*x - 1)
    #     ])

    ##### LAUNCHING COMPUTATION FOR VAE #####

    # Loading the VAE model
    model = model_type(latent_dimension)
    model.load_state_dict(torch.load(VAE_nn_saving_path, map_location='cpu'))
    model.training = False
    model = model.to(device)    

    dataset = Dataset2D_patch("data_csv/starmen_patch_test_set.csv", read_image=open_npy, transform=transformations)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_worker, shuffle=True, pin_memory=True, collate_fn=collate_2D_patch)
    all_losses = []

    # 1 epoch to get all reconstruction error with VAE
    num_image = 0
    with torch.no_grad():
        model.eval()
        print("Start computation with VAE")
        for x in data_loader:
            print("Image number", num_image)
            num_image += 1
            x = x.to(device)
            x = x.reshape(-1, 1, patch_size, patch_size)

            mu, logvar, recon_x, _ = model(x)
            reconstruction_loss = loss_function(recon_x, x)

            all_losses.extend(reconstruction_loss.tolist())

    stats_dict.update(compute_stats(all_losses, "VAE", method))

    plot_recon_error_histogram_patch(np.array(all_losses), "VAE", method)



    ##### LAUNCHING COMPUTATION FOR LVAE #####

    # # Loading the longitudinal model
    # model = model_type(latent_dimension)
    # model.load_state_dict(torch.load(LVAE_nn_saving_path, map_location='cpu'))
    # model.training = False
    # longitudinal_estimator = Leaspy.load(longitudinal_saving_path)
    # model = model.to(device)

    # if set_choice == "train":
    #     dataset = LongitudinalDataset2D_patch("data_csv/starmen_train_set.csv", read_image=open_npy, transform=transformations)
    # else:
    #     dataset = LongitudinalDataset2D_patch("data_csv/starmen_test_set.csv", read_image=open_npy, transform=transformations)
    # data_loader = DataLoader(dataset, batch_size=1, num_workers=num_worker, shuffle=False, collate_fn=longitudinal_collate_2D_patch)
    # all_losses = []

    # # 1 epoch to get all reconstruction error with LVAE
    # with torch.no_grad():
    #     for data in data_loader:
    #         x = data[0]
    #         mus, logvars, recon_x = get_longitudinal_images(data, model, longitudinal_estimator)
    #         for i in range(len(mus)):
    #             reconstruction_loss = loss_function(recon_x[i], x[i], method)
    #             if method == "pixel":
    #                 reconstruction_loss = reconstruction_loss.flatten()

    #             all_losses.append(reconstruction_loss)

    # stats_dict.update(compute_stats(all_losses, "LVAE", method))
    # if method != "pixel_all":
    #     plot_recon_error_histogram_patch(np.array(all_losses), "LVAE", method)


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
        
        # print("Stats for LVAE losses :")
        # print("min =", stats_dict["LVAE_min"])
        # print("max =", stats_dict["LVAE_max"])
        # print("mean =", stats_dict["LVAE_mean"])
        # print("median =", stats_dict["LVAE_median"])
        # print(f"Number of {method} above VAE_95 =", np.sum(all_losses > stats_dict["VAE_threshold_95"]))
        # print("95th percentile =", stats_dict["LVAE_threshold_95"])
        # print("99th percentile =", stats_dict["LVAE_threshold_99"])

        print("dict =", stats_dict) 

    # Saving the stats dictionnary and all the losses in a json file
    with open(f'data_csv/VAE_patch_losses_{latent_dimension}.json', 'w') as f:
        json.dump(all_losses, f)

    with open(f'data_csv/threshold_json/anomaly_threshold_patch_{method}_{latent_dimension}_{beta}.json', 'w') as f:
        json.dump(stats_dict, f, ensure_ascii=False)