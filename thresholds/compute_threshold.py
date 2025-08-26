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
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D

from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

from utils.display_individual_observations_2D import project_encodings_for_results, get_longitudinal_images
from utils.loading_image import open_npy

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

transformations = transforms.Compose([])


def compute_stats(all_losses, ssim, psnr, model, method):
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
        stats_dict[f"{model}_ssim"] = ssim.item()
        stats_dict[f"{model}_psnr"] = psnr.item()

    else: 
        stats_dict[f"{model}_threshold_95"] = np.percentile(all_losses, 95, axis=0).tolist()
        stats_dict[f"{model}_threshold_99"] = np.percentile(all_losses, 99, axis=0).tolist()
        stats_dict[f"{model}_median"] = np.median(all_losses, axis=0).tolist()
        stats_dict[f"{model}_min"] = np.min(all_losses, axis=0).tolist()
        stats_dict[f"{model}_max"] = np.max(all_losses, axis=0).tolist()
        stats_dict[f"{model}_mean"] = np.mean(all_losses, axis=0).tolist()
    return stats_dict


def plot_recon_error_histogram(recon_error_list, model_name, method):
    save_path = f"plots/recon_error/hist_{model_name}_{method}.pdf"
    os.makedirs(f"plots/recon_error/", exist_ok=True)
    color = "tab:orange" if "LVAE" in model_name else "tab:blue"

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
        ax.set_title(f'Reconstruction errors when considering {method} with {model_name}')
        if set_choice == "train":
            ax.set_ylim(0, 5000)  # Set the y-axis range to fit your expected scale
        else:
            ax.set_ylim(0, 1500)

    elif method == "pixel":
        ax.set_title(f'Pixel differences with {model_name}')
        if set_choice == "train":
            ax.set_ylim(0, 2.5e7)
        else:
            ax.set_ylim(0, 1e7)

    # Layout fix
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=False, default="pixel")
    parser.add_argument("-set", type=str, required=False, default="test")
    parser.add_argument("--dim", type=int, required=False, default=4)
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument("--gamma", type=float, required=False, default=100)
    args = parser.parse_args()

    method = args.method
    if method == "image":
        loss_function = image_reconstruction_error
    elif method == "pixel" or method == "pixel_all":
        loss_function = pixel_reconstruction_error
    else:
        print("Error in the input_loss, select one among the following : ['image', 'pixel', 'pixel_all]")
        exit()
    beta = args.beta
    gamma = args.gamma
    set_choice = args.set
    stats_dict = {}

    # Setting some parameters
    latent_dimension = args.dim
    num_worker = round(os.cpu_count()/6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    VAE_nn_saving_path = f"saved_models_2D/best_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    LVAE_nn_saving_path = f"saved_models_2D/best_fold_CVAE2D_{latent_dimension}_{beta}_{gamma}_200.pth2"
    longitudinal_saving_path = f"saved_models_2D/best_fold_longitudinal_estimator_params_CVAE2D_{latent_dimension}_{beta}_{gamma}_200.json2"

    ssim_metric_VAE = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric_VAE = PeakSignalNoiseRatio(data_range=1.0).to(device)

    ##### LAUNCHING COMPUTATION FOR VAE #####

    # Loading the VAE mode
    model = CVAE2D_ORIGINAL(latent_dimension)
    model.load_state_dict(torch.load(VAE_nn_saving_path, map_location='cpu'))
    model.to(device)
    model.training = False

    if set_choice == "train":
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
            ssim_metric_VAE.update(recon_x, x)
            psnr_metric_VAE.update(recon_x, x)

            loss = reconstruction_loss
            all_losses.append(loss)

        final_ssim_VAE = ssim_metric_VAE.compute()
        final_psnr_VAE = psnr_metric_VAE.compute()
    stats_dict.update(compute_stats(all_losses, final_ssim_VAE, final_psnr_VAE, "VAE", method))
    if method != "pixel_all":
        plot_recon_error_histogram(np.array(all_losses), f"VAE_{latent_dimension}_{beta}", method)



    ##### LAUNCHING COMPUTATION FOR LVAE #####

    # Loading the longitudinal model
    model = CVAE2D_ORIGINAL(latent_dimension)
    model.load_state_dict(torch.load(LVAE_nn_saving_path, map_location='cpu'))
    model.to(device)
    model.training = False
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path)

    ssim_metric_LVAE = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric_LVAE = PeakSignalNoiseRatio(data_range=1.0).to(device)

    if set_choice == "train":
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
            ssim_metric_LVAE.update(recon_x, x)
            psnr_metric_LVAE.update(recon_x, x)
            for i in range(len(mus)):
                reconstruction_loss = loss_function(recon_x[i], x[i], method)
                if method == "pixel":
                    reconstruction_loss = reconstruction_loss.flatten()

                all_losses.append(reconstruction_loss)

    final_ssim_LVAE = ssim_metric_LVAE.compute()
    final_psnr_LVAE = psnr_metric_LVAE.compute()
    stats_dict.update(compute_stats(all_losses, final_ssim_LVAE, final_psnr_LVAE, "LVAE", method))
    if method != "pixel_all":
        plot_recon_error_histogram(np.array(all_losses), f"LVAE_{latent_dimension}_{beta}_{gamma}", method)


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
        print("VAE SSIM =", final_ssim_VAE)
        print("VAE PSNR =", final_psnr_VAE)

        print()
        
        print("Stats for LVAE losses :")
        print("min =", stats_dict["LVAE_min"])
        print("max =", stats_dict["LVAE_max"])
        print("mean =", stats_dict["LVAE_mean"])
        print("median =", stats_dict["LVAE_median"])
        print(f"Number of {method} above VAE_95 =", np.sum(all_losses > stats_dict["VAE_threshold_95"]))
        print("95th percentile =", stats_dict["LVAE_threshold_95"])
        print("99th percentile =", stats_dict["LVAE_threshold_99"])
        print("LVAE SSIM =", final_ssim_LVAE)
        print("LVAE PSNR =", final_psnr_LVAE)

        print()

        print("dict =", stats_dict) 

    # Saving the stats dictionnary in a json file
    with open(f'data_csv/threshold_json/anomaly_threshold_{method}_{latent_dimension}_{beta}.json', 'w') as f:
        json.dump(stats_dict, f, ensure_ascii=False)
