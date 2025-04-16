import numpy as np
import pandas as pd
import torch
from leaspy import Leaspy
import matplotlib.pyplot as plt
import argparse
import os

from nnModels.CVAE2D import CVAE2D
from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from utils_display.display_individual_observations_2D import display_individual_observations_2D


if __name__=="__main__":

    # Parameters
    latent_representation_size = 4
    nn_saving_path = "saved_models_2D/CVAE2D_4_5_100_200.pth"
    longitudinal_saving_path = "saved_models_2D/longitudinal_estimator_params_CVAE2D_4_5_100_200.json"
    dataset_path = "../starmen_dataset.csv"
    df_dataset = pd.read_csv(dataset_path)
    output_path = "results_reconstruction/CVAE2D_4_5_100_200/"

    # Loading the models
    model = CVAE2D_ORIGINAL(latent_representation_size)
    model.load_state_dict(torch.load(nn_saving_path + "2", map_location='cpu'))
    saem_estimator = Leaspy.load(longitudinal_saving_path + "2")

    num_sample = 10
    random_patient = np.random.choice(df_dataset["subject_id"].unique(), num_sample, replace=False)
    print(random_patient)
    for patient_id in random_patient:
        display_individual_observations_2D(model, patient_id, dataset_path, 
                                           fitted_longitudinal_estimator=saem_estimator,
                                           save_path=f"{output_path}results_2D_subject{patient_id}_proj.pdf")
        display_individual_observations_2D(model, patient_id, dataset_path, 
                                           fitted_longitudinal_estimator=None,
                                           save_path=f"{output_path}results_2D_subject{patient_id}_noproj.pdf")


