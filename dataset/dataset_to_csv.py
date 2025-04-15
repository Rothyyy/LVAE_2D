import pandas as pd
import numpy as np

if __name__=="__main__":
    # Open and get all ages
    f = open("data_starmen/path_to_visit_ages_file.txt", "r")
    ages = f.read().split()

    # Process data to get a csv file 
    data = []
    for patient_id in range(1000):
        for i in range(10):
            row = {
                "age": ages[patient_id*10 + i] ,
                "image_path": f"./data_starmen/images/SimulatedData__Reconstruction__starman__subject_s{patient_id}__tp_{i}.npy" ,
                "subject_id": str(patient_id)
            }
            data.append(row)

    data_df = pd.DataFrame(data)
    data_df.to_csv("starmen_dataset.csv")




