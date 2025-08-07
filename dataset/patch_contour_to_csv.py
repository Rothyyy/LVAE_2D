import pandas as pd
import numpy as np
import argparse
import cv2
import os

def extract_centered_patches_from_contour(image, contour_mask, patch_size=15):
    """
    This functions takes as input an image (numpy array), the filled contour mask and the patch size.
    It returns the patches centered in every pixel of the filled contour and the coordinates of its center.
    """
    assert patch_size % 2 == 1, "Patch size must be odd."
    h, w = image.shape[:2]
    half = patch_size // 2

    num_patches = np.sum(contour_mask > 0)
    patches = np.zeros((num_patches, patch_size, patch_size))
    centers = np.zeros((num_patches, 2))
    patch_id = 0
    for i in range(half, h - half):
        for j in range(half, w - half):
            if contour_mask[i,j] > 0:
                patch = image[i - half: i + half + 1, j - half: j + half + 1] 
                patches[patch_id] = patch
                centers[patch_id] = [i, j]
                patch_id += 1

    return patches[:patch_id], centers[:patch_id].astype(int)

def get_filled_contour_mask(image):

    # Put image pixel value in [0, 255]
    if np.max(image <= 1):
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Threshold the image
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # OPTIONAL: Morph close to fill small black holes
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Threshold the image
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask (same size as image)
    mask = np.zeros_like(image)

    # Draw filled contour(s) on the mask
    cv2.drawContours(mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    # Put the image pixel value back to [0, 1]
    image = (image/255).astype(np.float64)

    return image, mask

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
    black_patch_threshold = np.ones((patch_size, patch_size)) * 1e-6

    for patient_id in range(1000):  # For every subject
        print("Patient :", patient_id)
        
        for t in range(10):     # For every timestamp
            
            # Load image and get all patches
            path_image = f"./data_starmen/images/SimulatedData__Reconstruction__starman__subject_s{patient_id}__tp_{t}.npy"
            image = np.load(path_image)
            image, mask = get_filled_contour_mask(image)
            patches, centers = extract_centered_patches_from_contour(image, mask, patch_size)

            # Store the information in a row
            np.save(f"./data_starmen/images_patch/Starman__subject_s{patient_id}__tp_{t}_patches.npy", patches)
            np.save(f"./data_starmen/images_patch/Starman__subject_s{patient_id}_tp_{t}_centers.npy", centers)
            row = {
                "subject_id": str(patient_id),
                "age": ages[patient_id*10 + t],
                "patch_path": f"./data_starmen/images_patch/Starman__subject_s{patient_id}__tp_{t}_patches.npy",
                "centers_path": f"./data_starmen/images_patch/Starman__subject_s{patient_id}_tp_{t}_centers.npy"
            }
            data.append(row)

        
    data_df = pd.DataFrame(data)
    data_df.to_csv("data_csv/starmen_dataset_patch.csv")


