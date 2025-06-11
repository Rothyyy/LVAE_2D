import cv2 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


# Drawing circles on some specific part of the image (left part of the starman, center of the starman)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_sample", type=int, required=False, default=5)
    parser.add_argument("-a", "--anomaly", type=str, required=False, default="darker_circle")
    parser.add_argument("--csv", type=bool, required=False, default=True)
    parser.add_argument("-set", type=str, required=False, default="test",
                        help="choose which set ('train', 'test', 'all') to consider when generating anomalous images.")
    args = parser.parse_args()

    anomaly = args.anomaly
    anomaly_list = ["darker_circle", "darker_line", "growing_circle", "shrinking_circle"]
    if anomaly not in anomaly_list:
        print("Error, anomaly not found, select one of the following anomaly : 'darker_circle', 'darker_line', 'growing_circle' ")
        exit()

    n_sample = args.n_sample      # Number of subject to generate an anomaly
    if args.set == "test":
        file_csv = pd.read_csv("data_csv/starmen_test_set.csv")
        id_list = file_csv["subject_id"].unique()
        random_subject = np.random.choice(id_list, size=n_sample, replace=False)
    
    elif args.set == "train":
        file_csv = pd.read_csv("data_csv/starmen_train_set.csv")
        id_list = file_csv["subject_id"].unique()
        random_subject = np.random.choice(id_list, size=n_sample, replace=False)

    else:
        random_subject = np.random.choice(1000, size=n_sample, replace=False)
    
    to_csv = args.csv
    data = []
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
            _, binary = cv2.threshold(image_uint8, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # This block is to put a circle on the most left part of the image (hopefully, the left hand)
            shape_contour = contours[0].shape
            contours = contours[0].reshape((shape_contour[0], shape_contour[-1]))
            leftmost_position = np.argmin(contours[:, 0])

            # Add the anomaly
            if anomaly == "growing_circle":
                cv2.circle(image_uint8, contours[leftmost_position], round(ages[t]-ages[0]), (255,255,255), -1)
            if anomaly == "shrinking_circle":
                cv2.circle(image_uint8, contours[leftmost_position], round(ages[9-t]-ages[0]), (255,255,255), -1)
            if anomaly == "darker_circle":
                cv2.circle(image_uint8, contours[leftmost_position], 2, 
                           (max(0, 200-20*round(ages[t]-ages[0])),
                            max(0, 200-20*round(ages[t]-ages[0])),
                            max(0, 200-20*round(ages[t]-ages[0]))), -1)
            if anomaly == "darker_line":
                cv2.line(image_uint8, contours[leftmost_position], contours[leftmost_position] + 3,     # TODO: Maybe consider better line position ?
                         ((max(0, 200-20*round(ages[t]-ages[0])),
                           max(0, 200-20*round(ages[t]-ages[0])),
                           max(0, 200-20*round(ages[t]-ages[0])))), 2)

            # Create the data that will be used for anomaly detection
            if to_csv:
                row = {
                "age": ages[t] , 
                "image_path": f"./data_starmen/anomaly_images/{anomaly_image_name}" ,
                "subject_id": str(subject)
            }
                data.append(row)

            # Saving the image in npy format
            image = image_uint8.astype(np.float64) / 255
            np.save(save_path, image)

    # Saving the data in csv file
    data_df = pd.DataFrame(data)
    data_df.to_csv(f"data_csv/anomaly_{anomaly}_starmen_dataset.csv")

