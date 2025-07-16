import pandas as pd
import numpy as np
import argparse
import os

def extract_centered_patches(image, patch_size=5):
    """
    This functions takes as input an image (numpy array) and the patch size.
    It returns the patches centered in every pixel not in the border and the coordinates of its center.
    """
    assert patch_size % 2 == 1, "Patch size must be odd."
    h, w = image.shape[:2]
    half = patch_size // 2

    patches = []
    centers = []

    for i in range(half, h - half):
        for j in range(half, w - half):
            patch = image[i - half:i + half + 1, j - half:j + half + 1]  # (h, w)
            patches.append(patch)
            centers.append((i, j))

    return np.array(patches), centers

def get_patch_centers_for_pixel(pixel_coord, image_shape, patch_size):
    """
    For a given pixel coordinate, image shape and patch size.
    """
    i, j = pixel_coord
    h, w = image_shape
    half = patch_size // 2

    # Valid center range for patches
    min_i = max(half, i - half)
    max_i = min(h - half - 1, i + half)
    min_j = max(half, j - half)
    max_j = min(w - half - 1, j + half)

    centers = [(ci, cj) 
               for ci in range(min_i, max_i + 1)
               for cj in range(min_j, max_j + 1)]
    
    return centers


def get_patch_centers(patch_size=15, image_shape=(64,64)):
    """
    This function takes the patch size and the image shape and returns all the patch's centers.
    """
    assert patch_size % 2 == 1, "Patch size must be odd."
    h, w = image_shape[:2]
    half = patch_size // 2

    # Valid center coordinates (avoid borders)
    y_coords = np.arange(half, h - half)
    x_coords = np.arange(half, w - half)

    # Create grid of centers
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    centers = np.stack([yy.ravel(), xx.ravel()], axis=1)

    return centers  # shape: (N, 2)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=int, required=False, default=15)
    args = parser.parse_args()

    patch_size = args.size
    num_patch = (64 - (patch_size//2 * 2)) * (64 - (patch_size//2 * 2))     # Number of patches per image
    # Open and get all ages
    f = open("data_starmen/path_to_visit_ages_file.txt", "r")
    ages = f.read().split()

    os.makedirs("./data_starmen/images_patch", exist_ok=True)

    # Process data to get a csv file 
    data = []
    patch_id = 0
    for patient_id in range(1000):  # For every subject
        print("Patient :", patient_id)
        for t in range(10):     # For every timestamp
            
            # Load image and get all patches
            path_image = f"./data_starmen/images/SimulatedData__Reconstruction__starman__subject_s{patient_id}__tp_{t}.npy"
            image = np.load(path_image) 
            # patches, centers = extract_centered_patches(image, patch_size)
            # # Store the information in a row
            # np.save(f"./data_starmen/images_patch/Starman__subject_s{patient_id}__tp_{t}_patches.npy", patches)
            
            row = {
                "subject_id": str(patient_id),
                "patch_id_min": patch_id,
                "patch_id_max": patch_id + num_patch - 1,
                "age": ages[patient_id*10 + t],
                "patch_path": f"./data_starmen/images_patch/Starman__subject_s{patient_id}__tp_{t}_patches.npy" ,
            }
            data.append(row)

        patch_id += num_patch
            

    data_df = pd.DataFrame(data)
    data_df.to_csv("data_csv/starmen_dataset_patch.csv")


