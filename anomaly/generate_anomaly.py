import cv2 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

from dataset.patch_to_csv import extract_centered_patches
from dataset.patch_contour_to_csv import extract_centered_patches_from_contour, get_filled_contour_mask


# Drawing circles on some specific part of the image (left part of the starman, center of the starman)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_sample", type=int, required=False, default=5)
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="darker_line")
    parser.add_argument("-p", "--patch", type=bool, required=False, default=False,
                        help="Extract patches and create csv file ?")
    parser.add_argument("-pc", "--patch_contour", type=bool, required=False, default=False,
                        help="Extract patches from the contour and create csv file ?")
    parser.add_argument("-s", "--size", type=int, required=False, default=15,
                        help="patch size")
    args = parser.parse_args()

    anomaly = args.anomaly
    anomaly_list = ["darker_circle", "darker_line", "growing_circle", "shrinking_circle"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'darker_circle', 'darker_line', 'growing_circle' ")
        exit()

    n_sample = args.n_sample      # Number of subject to generate an anomaly
    file_csv = pd.read_csv("data_csv/starmen_test_set.csv")
    id_list = file_csv["subject_id"].unique()
    random_subject = np.random.choice(id_list, size=n_sample, replace=False)
    
    
    get_patch = args.patch
    get_patch_contour = args.patch_contour
    os.makedirs("data_starmen/anomaly_patches", exist_ok=True)
    patch_size = args.size
    num_patch = (64 - (patch_size//2 * 2)) * (64 - (patch_size//2 * 2))     # Number of patches per image

    data_image = []
    data_patches = []
    f = open("data_starmen/path_to_visit_ages_file.txt", "r")
    ages_list = f.read().split()

    for subject in random_subject:
        ages = np.array(ages_list[subject*10 : subject*10+10]).astype(float)
        for t in range(10):
            
            image_path = f"data_starmen/images/SimulatedData__Reconstruction__starman__subject_s{subject}__tp_{t}.npy"
            anomaly_image_name = f"{anomaly}__starman__subject_s{subject}__tp_{t}.npy"
            save_path = f"data_starmen/anomaly_images/{anomaly_image_name}"
            image = np.load(image_path)

            # Scale from [0, 1] to [0, 255] and convert to uint8
            image_uint8 = (image * 255).astype(np.uint8)

            # Optional: threshold the image to get binary input for contours
            _, binary = cv2.threshold(image_uint8, 80, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # This block is to get the coordinates of the most left part of the image (hopefully, the left hand)
            shape_contour = contours[0].shape
            contours = contours[0].reshape((shape_contour[0], shape_contour[-1]))
            leftmost_position = np.argmin(contours[:, 0])

            # Add the anomaly
            if anomaly == "growing_circle":
                cv2.circle(image_uint8, contours[leftmost_position], round(ages[t]-ages[0]), (255,255,255), -1)
            if anomaly == "shrinking_circle":
                cv2.circle(image_uint8, contours[leftmost_position], round(ages[9-t]-ages[0]), (255,255,255), -1)
            if anomaly == "darker_circle":
                cv2.circle(image_uint8, contours[leftmost_position] + [3,1], 2, 
                           (max(0, 200-20*round(ages[t]-ages[0])),
                            max(0, 200-20*round(ages[t]-ages[0])),
                            max(0, 200-20*round(ages[t]-ages[0]))), -1)
            if anomaly == "darker_line":
                cv2.line(image_uint8, contours[leftmost_position] + [3, 1], contours[leftmost_position] + [5 , 2],     # TODO: Maybe consider better line position ?
                         ((max(0, 200-20*round(ages[t]-ages[0])),
                           max(0, 200-20*round(ages[t]-ages[0])),
                           max(0, 200-20*round(ages[t]-ages[0])))), 2)


            # Saving the image in npy format
            image = image_uint8.astype(np.float64) / 255
            np.save(save_path, image)

            # Create the data that will be used for anomaly detection
            row = {
                "age": ages[t] , 
                "image_path": f"./data_starmen/anomaly_images/{anomaly_image_name}" ,
                "subject_id": str(subject) 
            }
            data_image.append(row)

            if get_patch:
                patches, _ = extract_centered_patches(image, patch_size=patch_size)
                patch_id = (subject) * 2500 
                f"{anomaly}__starman__subject_s{subject}__tp_{t}.npy"
                np.save(f"./data_starmen/anomaly_patches/{anomaly}__starman__subject_s{subject}__tp_{t}_patches.npy", patches)
                row = {
                    "subject_id": str(subject),
                    "patch_id_min": patch_id,
                    "patch_id_max": patch_id + num_patch - 1,
                    "age": ages[t],
                    "patch_path": f"./data_starmen/anomaly_patches/{anomaly}__starman__subject_s{subject}__tp_{t}_patches.npy" ,
                }
                data_patches.append(row)

            if get_patch_contour:
                image, mask = get_filled_contour_mask(image)
                patches, centers = extract_centered_patches_from_contour(image, mask, patch_size)

                # Store the information in a row
                np.save(f"./data_starmen/anomaly_patches/Starman__subject_s{subject}__tp_{t}_patches.npy", patches)
                np.save(f"./data_starmen/anomaly_patches/Starman__subject_s{subject}_tp_{t}_centers.npy", centers)
                row = {
                    "subject_id": str(subject),
                    "age": ages[t],
                    "patch_path": f"./data_starmen/anomaly_patches/Starman__subject_s{subject}__tp_{t}_patches.npy",
                    "centers_path": f"./data_starmen/anomaly_patches/Starman__subject_s{subject}_tp_{t}_centers.npy"
                }
                data_patches.append(row)


    # Saving the data in csv file
    data_df_image = pd.DataFrame(data_image)
    data_df_image.to_csv(f"data_csv/anomaly_{anomaly}_starmen_dataset.csv")

    if get_patch or get_patch_contour:
        data_df_patch = pd.DataFrame(data_patches)
        data_df_patch.to_csv(f"data_csv/anomaly_{anomaly}_starmen_dataset_patch.csv")

