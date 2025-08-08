import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_anomaly_bar(array_anomaly_detected, model_name, anomaly_type, method, num_images):
    """
    This function will plot bars corresponding to the number of time the model detect
    an anomaly for the i-th image of a subject.
    """
    save_path = f"plots/fig_anomaly_reconstruction/{anomaly_type}/{model_name}_{method}_{anomaly_type}_bar_plot.pdf"
    os.makedirs(f"plots/fig_anomaly_reconstruction/{anomaly_type}", exist_ok=True)
    x = np.array([i for i in range(1, 11)])
    color = "tab:blue" if "VAE" in model_name else "tab:orange"

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


def plot_anomaly_figure(original_image, reconstructed_image_VAE, reconstructed_image_LVAE,
                  id, anomaly_type, method, 
                  detection_vector_VAE, detection_vector_LVAE, pixel_anomaly_VAE=None, pixel_anomaly_LVAE=None):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """
    os.makedirs(f"plots/fig_anomaly_reconstruction/{anomaly_type}/{method}", exist_ok=True)
    save_path = f"plots/fig_anomaly_reconstruction/{anomaly_type}/{method}/AD_subject_{id}.pdf"

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
        axarr[0, i].axis('off')

        axarr[1, i].imshow(reconstructed_image_VAE[i, 0, :, :], cmap="gray")
        axarr[1, i].axis('off')

        axarr[2, i].imshow(reconstructed_image_LVAE[i, 0, :, :], cmap="gray")
        axarr[2, i].axis('off')

        axarr[3, i].imshow(binary_overlay[i])
        axarr[3, i].axis('off')

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


def plot_anomaly_figure_patch(original_image, reconstructed_image_VAE, anomaly_map_VAE,
                                id, anomaly_type, latent_dimension = None):
    """
    We enter this function when an anomaly is detected.
    The function will plot the image and save it in a pdf file.
    """
    os.makedirs(f"plots/fig_anomaly_reconstruction/{anomaly_type}", exist_ok=True)
    save_path = f"plots/fig_anomaly_reconstruction/{anomaly_type}/AD_{latent_dimension}_subject_{id}.pdf"

    # Compute the residual and binary mask
    fig_width = original_image.shape[0] * 10
    fig_height = 50  # Adjust as needed

    # To get RGB output in the anomaly map
    binary_overlay = np.zeros((10,64,64,3))
    binary_overlay[:,:,:, 2] = anomaly_map_VAE
    f, axarr = plt.subplots(3, 10, figsize=(fig_width, fig_height))
    for i in range(original_image.shape[0]):
        axarr[0, i].imshow(original_image[i,:, :], cmap="gray")
        axarr[0, i].axis('off')

        axarr[1, i].imshow(reconstructed_image_VAE[i, :, :], cmap="gray")
        axarr[1, i].axis('off')

        axarr[2, i].imshow(binary_overlay[i, :, :])
        axarr[2, i].axis('off')

    # Row labels
    row_labels = ["Input", "VAE", "Anomalies"]
    for row in range(3):
        # Add label to the first column of each row, closer and vertically centered
        axarr[row, 0].annotate(row_labels[row],
                            xy=(-0.1, 0.5),  # Slightly to the left, centered vertically
                            xycoords='axes fraction',
                            ha='right',
                            va='center',
                            fontsize=60)

    f.suptitle(f'Individual id = {id}', fontsize=80)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(f)
    plt.clf()
    return 


def plot_anomaly_figure_patch_heatmap(inputs, reconstructions, anomaly_map_VAE, anomaly_scores, anomaly_type, id, latent_dimension=64, cmap='jet'):
    """
    Plots a 3-row grid of:
    - original images
    - reconstructions
    - anomaly heatmaps
    
    Parameters:
        inputs (np.ndarray): Array of shape (N, H, W) or (N, H, W, 3)
        recons (np.ndarray): Array of same shape as inputs
        anomaly_scores (np.ndarray): Array of shape (N, H, W)
        cmap (str): Colormap for the heatmaps
    """

    # os.makedirs(f"plots/fig_anomaly_reconstruction/{anomaly_type}", exist_ok=True)
    # save_path = f"plots/fig_anomaly_reconstruction/{anomaly_type}/AD_{latent_dimension}_subject_{id}_HM.pdf"
    os.makedirs(f"plots/fig_anomaly_reconstruction/{anomaly_type}", exist_ok=True)
    save_path = f"plots/fig_anomaly_reconstruction/{anomaly_type}/AD_{latent_dimension}_subject_{id}.pdf"

    N = anomaly_scores.shape[0]
    # Compute the residual and binary mask
    fig_width = inputs.shape[0] * 10
    fig_height = 50  # Adjust as needed

    fig, axes = plt.subplots(4, N, figsize=(fig_width, fig_height), constrained_layout=True)  # 4 rows, N columns

    # To get RGB output in the anomaly map
    binary_overlay = np.zeros((10,64,64,3))
    binary_overlay[:,:,:, 2] = anomaly_map_VAE

    vmin = np.min(anomaly_scores)
    vmax = np.max(anomaly_scores)

    for i in range(N):
        # --- Row 1: Input ---
        axes[0, i].imshow(inputs[i], cmap='gray')
        axes[0, i].axis('off')

        # --- Row 2: Reconstruction ---
        axes[1, i].imshow(reconstructions[i], cmap='gray')
        axes[1, i].axis('off')

        # --- Row 3: Binary anomaly map ---
        axes[2, i].imshow(binary_overlay[i])
        axes[2, i].axis('off')

        # --- Row 4: Heatmap ---
        im = axes[3, i].imshow(anomaly_scores[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[3, i].axis('off')


    # Row labels
    row_labels = ["Input", "VAE", "Anomalies", "Heatmap"]
    for row in range(len(row_labels)):
        # Add label to the first column of each row, closer and vertically centered
        axes[row, 0].annotate(row_labels[row],
                            xy=(-0.1, 0.5),  # Slightly to the left, centered vertically
                            xycoords='axes fraction',
                            ha='right',
                            va='center',
                            fontsize=60)

    # fig.colorbar(im, ax=axes, orientation='vertical', label='Anomaly Score')
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
    cbar.set_label('Anomaly Score', fontsize=60)
    cbar.ax.tick_params(labelsize=48)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return 


