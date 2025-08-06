from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from nnModels.CVAE2D import CVAE2D
from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from leaspy import Leaspy, AlgorithmSettings

import pandas as pd

from utils.project_encodings_for_results import project_encodings_for_results
from utils.loading_image import open_npy
import os


device = "cuda" if torch.cuda.is_available() else "cpu"







def get_longitudinal_patches(data, model, fitted_longitudinal_estimator):
    encodings = []
    times = []
    patch_ids = []

    time_patch = data[1]
    subject_ids = data[2]
    centers_list = data[3]
    batch_size = len(centers_list)
    
    projection_timepoints = {}

    encoder_output = model.encoder(data[0].float().to(device))
    logvars = encoder_output[1].detach().clone().to(device)
    encodings.append((encoder_output[0].view((encoder_output[0].size(0), encoder_output[0].size(1)))).detach().clone().to(device))
    for i in range(batch_size):
        for t in range(10):     # Should be in range(len(centers_list[i]))  In case we do not have 10 images
            number_patches = len(centers_list[i][t])
            times.extend([time_patch[i][t]] * number_patches)
            for patch in range(number_patches):
                center_coord_x, center_coord_y = centers_list[i][t][patch]
                patch_id = f"s_{subject_ids[i]}_{int(center_coord_x)}_{int(center_coord_y)}"
                patch_ids.append(patch_id)
                projection_timepoints[patch_id] = times[-1]

    encodings = torch.cat(encodings)
    encodings_df = pd.DataFrame({'ID': patch_ids, 'TIME': times})
    for i in range(encodings.shape[1]):
        encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                            encodings[:, i].detach().clone().tolist())
    encodings_df['ID'] = encodings_df['ID'].astype(str)

    predicted_latent_variables, _ = project_encodings_for_results(encodings_df, None,
                                                                  fitted_longitudinal_estimator,
                                                                  projection_timepoints)
    projected_patches = model.decoder(torch.tensor(predicted_latent_variables[str(subject_id)]).to(device))
    # return encodings, logvars, projected_patches
    return torch.from_numpy(predicted_latent_variables[str(subject_id)]), logvars, projected_patches


def get_longitudinal_images(data, model, fitted_longitudinal_estimator):
    encodings = []
    times = []
    ids = []
    subject_id = data[2][0]

    encoder_output = model.encoder(data[0].float().to(device))
    logvars = encoder_output[1].detach().clone().to(device)
    encodings.append(encoder_output[0].detach().clone().to(device))
    for i in range(len(data[1])):
        times.extend(data[1][i])
        ids.extend([data[2][i]] * len(data[1][i]))
    encodings = torch.cat(encodings)
    encodings_df = pd.DataFrame({'ID': ids, 'TIME': times})
    for i in range(encodings.shape[1]):
        encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                            encodings[:, i].detach().clone().tolist())
    encodings_df['ID'] = encodings_df['ID'].astype(str)
    projection_timepoints = {str(subject_id): data[1][0]}
    predicted_latent_variables, _ = project_encodings_for_results(encodings_df, str(subject_id),
                                                                                fitted_longitudinal_estimator,
                                                                                projection_timepoints)
    projected_images = model.decoder(torch.tensor(predicted_latent_variables[str(subject_id)]).to(device))
    # return encodings, logvars, projected_images
    return torch.from_numpy(predicted_latent_variables[str(subject_id)]), logvars, projected_images


def display_individual_observations_2D(model, subject_id, dataset_csv, transformations=None,
                                               fitted_longitudinal_estimator=None, save_path="results.pdf",
                                               device='cuda' if torch.cuda.is_available() else 'cpu', ):
    """
    Displays the inference of the trained model on a subject's observations data and saves it in a file.
    The display is an image with 2 rows one with the three patients observations and the bottom one is formed by the first image
    reconstructed and then projected by the models at the times of the inputs.
    If no longitudinal estimator is given then it simplpy displays 2 reconstructions from the VAE.
    :args: model: trained VAE model
    :args: subject_id: id of the subject you want to test the model on
    :args: dataset_csv: path to the csv file containing the information mus shaped like all the other data csv file, see implementation
    of IndividualLongitudinalDataset for more info
    :args: fitted_longitudinal_estimator: trained longitudinal mixed effects model
    :args: results_saving_path: path on which
    returns: None
    """

    class IndividualLongitudinalDataset(Dataset):
        def __init__(self, summary_file, subject_id, transform=None, target_transform=None,
                     read_image=lambda x: torch.Tensor(plt.imshow(x, cmap="gray"))):
            self.summary_dataframe = pd.read_csv(summary_file).sort_values(['subject_id', 'age'])
            self.subject_id = subject_id
            self.transform = transform
            self.target_transform = target_transform
            self.read_image = read_image
            self.list_patient_ids = self.summary_dataframe['subject_id'].unique().tolist()

        def __len__(self):
            return len(self.list_patient_ids)

        def __getitem__(self, idx):
            """

            returns observations for an individual, the time of observation,the id of the individual
            images.shape = number_of_observations x 1 x Depth x Height x Width
            """
            patient_id = self.subject_id
            summary_rows = self.summary_dataframe[self.summary_dataframe['subject_id'] == patient_id].sort_values(
                ['subject_id', 'age'])
            images = [self.read_image(summary_rows.iloc[i]['image_path']) for i in range(len(summary_rows))]
            if len(images) == 0:
                return None, None, None

            if self.transform is None:
                images = torch.stack(images)
            if self.transform:
                # print(patient_id)
                images = self.transform(torch.stack(images))

            if self.target_transform:
                label = self.target_transform(summary_rows)
                return NotImplemented

            return images, [summary_rows['age'].iloc[i] for i in range(len(summary_rows))], patient_id

    def my_collate(batch, device='cuda' if torch.cuda.is_available() else 'cpu'):
        images = torch.cat([item[0].to(device) for item in batch if item[0] is not None], axis=0)
        images = images.unsqueeze(1)
        infos = [item[1] for item in batch if item[0] is not None]
        ids = [item[2] for item in batch if item[0] is not None]

        return [images, infos, ids]

    individual_dataset = IndividualLongitudinalDataset(dataset_csv, subject_id, read_image=open_npy,
                                                       transform=transformations)
    data_loader = DataLoader(individual_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)
    data = next(iter(data_loader))
    model.to(device)
    model.training = False
    model.eval()
    total_number_of_observation = len(data[1][0])
    fig_width = total_number_of_observation * 10  # Adjust multiplier as needed
    fig_height = 50  # Adjust as needed
    with torch.no_grad():
        if fitted_longitudinal_estimator is not None:
            encodings = []
            times = []
            ids = []

            encodings.append(model.encoder(data[0].float().to(device))[
                                 0].detach().clone().to(device))
            for i in range(len(data[1])):
                times.extend(data[1][i])
                ids.extend([data[2][i]] * len(data[1][i]))
            encodings = torch.cat(encodings)
            encodings_df = pd.DataFrame({'ID': ids, 'TIME': times})
            for i in range(encodings.shape[1]):
                encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                                    encodings[:, i].detach().clone().tolist())
            encodings_df['ID'] = encodings_df['ID'].astype(str)
            projection_timepoints = {str(subject_id): data[1][0]}
            predicted_latent_variables, reference_times = project_encodings_for_results(encodings_df, str(subject_id),
                                                                                        fitted_longitudinal_estimator,
                                                                                        projection_timepoints)
            projected_images = model.decoder(torch.tensor(predicted_latent_variables[str(subject_id)]).to(device))
            f, axarr = plt.subplots(3, total_number_of_observation, figsize=(fig_width, fig_height))
            # print("Projected images :")
            # print(torch.unique(projected_images[2]))
            # print(projected_images[2].max())
            # print(projected_images[2].min())
            for i in range(len(data[1][0])):

                if data[1][0][i] in reference_times:
                    axarr[1, i].imshow(projected_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
                axarr[0, i].imshow(data[0][i].squeeze().cpu().detach().numpy(), cmap='gray')
                axarr[2, i].imshow(projected_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
                axarr[0, i].set_title(f"Age = {data[1][0][i]:.3f}", fontsize=64)

        else:
            f, axarr = plt.subplots(2, total_number_of_observation, figsize=(fig_width, fig_height))
            mu, logVar, reconstructed, encoded = model.forward(data[0].float().to(device))
            for i in range(len(data[1][0])):
                axarr[0, i].imshow(data[0][i].squeeze().cpu().detach().numpy(), cmap='gray')
                axarr[1, i].imshow(reconstructed[i].squeeze().cpu().detach().numpy(), cmap='gray')
                axarr[0, i].set_title(f"Age = {data[1][0][i]:.3f}", fontsize=64)

    f.suptitle(f'Individual id = {data[2][0]}', fontsize=80)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":

    nn_model_path = "saved_models_2D/original_clean_4_5.0_100.0_200.pth"
    longitudinal_estimator_path = "saved_models_2D/longitudinal_estimator_params_original_clean_4_5.0_100.0_200.json" #"saved_models_2D/longitudinal_estimator_params_final_model_2D_4_5.0_100.0_200.kk3"
    must_fit_longitudinal = True
    subject_id = 9
    model = CVAE2D_ORIGINAL(4, device='cpu', )
    model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
    model.eval()
    transformations = transforms.Compose([])

    easy_dataset = LongitudinalDataset2D('cleaned_starmen_csv.csv', read_image=open_npy,
                                         transform=transformations)
    data_loader = DataLoader(easy_dataset, batch_size=15, num_workers=0, shuffle=False,
                             collate_fn=longitudinal_collate_2D)
    test_saem_estimator = None

    if longitudinal_estimator_path is not None:
        test_saem_estimator = Leaspy.load(longitudinal_estimator_path)
        if must_fit_longitudinal:
            algo_settings = AlgorithmSettings('mcmc_saem', n_iter=30000, seed=45, noise_model="gaussian_diagonal")
            test_saem_estimator = Leaspy("linear", noise_model="gaussian_diagonal", source_dimension=3)
            test_saem_estimator, _ = fit_longitudinal_estimator_on_nn(data_loader, model, "cpu", test_saem_estimator,
                                                                      algo_settings)
            test_saem_estimator.save(longitudinal_estimator_path + "4")
            # test_saem_estimator = Leaspy.load(longitudinal_estimator_path)
    display_individual_observations_2D(model, 9, 'cleaned_starmen_csv.csv',
                                               fitted_longitudinal_estimator=test_saem_estimator,
                                               save_path=f"result/results++_2D_subject{subject_id}_proj_no_coupling.pdf")
