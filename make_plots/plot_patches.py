import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset

from nnModels.CVAE2D_PATCH import CVAE2D_PATCH
from dataset.Dataset2D import Dataset2D_patch
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch


def plot_patch_comparison(patches_original, patches_VAE, id=0):
    """
    This function takes 2 sequences of patches to plot in 2 rows
    """
    n_patches = patches_original.shape[0]
    fig, axes = plt.subplots(2, n_patches, figsize=(n_patches* 1.5, 4))  # 2 rows, n columns
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n_patches):
        # Upper row (patches_a)
        axes[0, i].imshow(patches_original[i], cmap='gray')
        axes[0, i].axis('off')

        # Bottom row (patches_b)
        axes[1, i].imshow(patches_VAE[i], cmap='gray')
        axes[1, i].axis('off')

    # Row labels
    row_labels = ["Original", "VAE"]
    for row in range(2):
        # Add label to the first column of each row, closer and vertically centered
        axes[row, 0].annotate(row_labels[row],
                            xy=(-0.1, 0.5),  # Slightly to the left, centered vertically
                            xycoords='axes fraction',
                            ha='right',
                            va='center',
                            fontsize=14)

    plot_filename = f"plots/VAE_patches/VAE_patch_plot_{id}.pdf"
    os.makedirs("plots/VAE_patches", exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_filename)
    # plt.show(fig)
    plt.clf()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", "--dim", type=int, required=False, default=64)
    parser.add_argument("--beta", type=float, required=False, default=2)
    args = parser.parse_args()

    latent_dimension = args.dimension
    beta = args.beta
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_VAE_path = f"saved_models_2D/best_patch_fold_CVAE2D_{latent_dimension}_{beta}.pth"
    model = CVAE2D_PATCH(latent_dimension)
    model.load_state_dict(torch.load(model_VAE_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    model.training = False
    model.to(device)

    ## VAE PART
    dataset = Dataset2D_patch("./data_csv/train_patch_folds/starmen_patch_train_set_fold_0.csv")
    dataloader = DataLoader(dataset, 10, num_workers=int(os.cpu_count()/4), shuffle=True, pin_memory=True)
    random_patches = np.random.choice(1000, size=10, replace=False)
    print("Random patch number =", random_patches)

    for data in dataloader:
        data = data.reshape(-1, 1, 15, 15)
        output = model(data)
        break

    plot_patch_comparison(data[random_patches[:5], 0].detach(), output[2][random_patches[:5], 0].detach(), 0)
    plot_patch_comparison(data[random_patches[5:], 0].detach(), output[2][random_patches[5:], 0].detach(), 1)


