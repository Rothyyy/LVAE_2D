import numpy as np
import pandas as pd
import torch
from leaspy import AlgorithmSettings, Leaspy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import os

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import spatial_auto_encoder_loss, longitudinal_loss

from dataset.Dataset2D import Dataset2D
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from longitudinalModel.project_encodings_for_training import project_encodings_for_training
from utils_display.project_encodings_for_results import project_encodings_for_results


def get_longitudinal_images(data, model, fitted_longitudinal_estimator, 
                            device="cuda" if torch.cuda.is_available() else "cpu"):
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


def CV_VAE(model_type, fold_index_list, test_set, nn_saving_path, dataset_name,
           device='cuda' if torch.cuda.is_available() else 'cpu', plot_save_path=None,
           latent_dimension=4, gamma=100, beta=5, freeze="no_freeze",
           batch_size=256, num_worker=round(os.cpu_count()/4)):

    best_fold = 0
    best_loss = torch.inf
    folds_test_loss = np.zeros(len(fold_index_list))
    dataset = Dataset2D(test_set)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True, pin_memory=True)

    for fold_index in range(len(fold_index_list)):
        model = model_type(latent_dimension)
        model.gamma = gamma
        model.beta = beta
        model.load_state_dict(torch.load(nn_saving_path+f"_fold_{fold_index}.pth", map_location='cpu'))

        model.device = device
        model.to(device)
        model.eval()
        model.training=False
        losses = []


        for x in data_loader:
            x = x.to(device)

            mu, logvar, recon_x, _ = model(x)
            reconstruction_loss, kl_loss = spatial_auto_encoder_loss(mu, logvar, recon_x, x)
            loss = reconstruction_loss + kl_loss * model.beta
            losses.append(loss)

        test_mean_loss = sum(losses) / len(losses)
        folds_test_loss[fold_index] = test_mean_loss.detach().to("cpu")

        if test_mean_loss < best_loss:
            best_loss = test_mean_loss
            best_fold = fold_index

    if plot_save_path is None:
        plot_save_path = f"training_plots/dataset_{dataset_name}/{freeze}/folds/CVAE2D_{args.dimension}_{args.beta}_{args.gamma}_{args.iterations}/CV_VAE_{freeze}_results.pdf"
    else:
        plot_save_path += f"CV_VAE_results.pdf"
    plt.plot(fold_index_list, folds_test_loss)
    plt.savefig(plot_save_path)
    plt.clf()

    return best_fold

def CV_LVAE(model_type, fold_index_list, test_set, nn_saving_path, longitudinal_saving_path, dataset_name,
           device='cuda' if torch.cuda.is_available() else 'cpu', freeze="no_freeze", plot_save_path=None,
           latent_dimension=4, gamma=100, beta=5,
           num_worker=round(os.cpu_count()/4)):

    dataset = LongitudinalDataset2D(test_set)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=num_worker, shuffle=True, pin_memory=True, collate_fn=longitudinal_collate_2D)
    best_fold = 0
    best_loss = torch.inf
    folds_test_loss = np.zeros(len(fold_index_list))

    for fold_index in range(len(fold_index_list)):
        model = model_type(latent_dimension)
        model.gamma = gamma
        model.beta = beta
        model.load_state_dict(torch.load(nn_saving_path+f"_fold_{fold_index}.pth2", map_location='cpu'))

        model.device = device
        model.to(device)
        model.eval()
        model.training=False

        longitudinal_estimator = Leaspy.load(longitudinal_saving_path+f"_fold_{fold_index}.json2")
        losses = [] 

        for data in test_data_loader:
            x = data[0].to(device).float()
            mus, logvars, recon_x = get_longitudinal_images(data, model, longitudinal_estimator)
            reconstruction_loss, kl_loss = spatial_auto_encoder_loss(mus, logvars, recon_x, x)

            loss = reconstruction_loss + model.beta * kl_loss       # Add alignment loss for longitudinal aspect ?     
            losses.append(loss)

        test_mean_loss = sum(losses) / len(losses)
        folds_test_loss[fold_index] = test_mean_loss.detach().to("cpu")

        if test_mean_loss < best_loss:
            best_loss = test_mean_loss
            best_fold = fold_index

    if plot_save_path is None:
        plot_save_path = f"training_plots/dataset_{dataset_name}/{freeze}/folds/CVAE2D_{args.dimension}_{args.beta}_{args.gamma}_{args.iterations}/CV_LVAE_{freeze}_results.pdf"
    else:
        plot_save_path += f"CV_LVAE_{freeze}_results.pdf"
    plt.plot(fold_index_list, folds_test_loss)
    plt.savefig(plot_save_path)
    plt.clf()

    return best_fold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=False, default=100,
                        help='hyperparameter gamma value used for computing the loss')
    parser.add_argument('--beta', type=float, required=False, default=5,
                        help='hyperparameter beta value used for computing the loss, default = 5')
    parser.add_argument('--dimension', type=int, required=False, default=4,
                        help='size of the latent representation generated by the neural network encoder, default =4')
    parser.add_argument('--batch_size', type=int, required=False, default=256,
                        help='batch_size to train the VAE, default = 256')
    parser.add_argument('--iterations', type=int, required=False, default=200,
                    help='Number of iterations when training the longitudinal estimator, default = 200')
    parser.add_argument('-f', '--freeze', type=str, required=True, default='y',
                        help='freeze convolution layer ? default = y')
    parser.add_argument('--dataset', type=str, required=True, default="noacc",
                        help='Use the models trained on dataset "acc" or "noacc"')
    temp_args, _ = parser.parse_known_args()

    freeze_path = "freeze_conv" if temp_args.freeze == 'y' else "no_freeze"

    parser.add_argument('--nnmodel_path', type=str, required=False,
                        default=f'saved_models_2D/dataset_{temp_args.dataset}/{freeze_path}/folds/CVAE2D_{temp_args.dimension}_{temp_args.beta}_{temp_args.gamma}_{temp_args.iterations}',
                        help='path where the neural network model parameters are saved')
    parser.add_argument('--longitudinal_estimator_path', type=str, required=False,
                        default=f'saved_models_2D/dataset_{temp_args.dataset}/{freeze_path}/folds/longitudinal_estimator_params_CVAE2D_{temp_args.dimension}_{temp_args.beta}_{temp_args.gamma}_{temp_args.iterations}',
                        help='path where the longitudinal estimator parameters are saved')
    args = parser.parse_args()

    test_set_path = "./data_csv/starmen_test_set.csv"
    test_set = pd.read_csv(test_set_path)
    nn_saving_path = args.nnmodel_path
    dataset_name = args.dataset
    longitudinal_saving_path = args.longitudinal_estimator_path
    batch_size = args.batch_size
    latent_representation_size = args.dimension
    gamma = args.gamma
    beta = args.beta

    save_best_fold_path_VAE = f"saved_models_2D/dataset_{dataset_name}/{freeze_path}/best_{freeze_path}_fold_CVAE2D_{args.dimension}_{args.beta}_{args.gamma}_{args.iterations}.pth"
    save_best_fold_path_LVAE = f"saved_models_2D/dataset_{dataset_name}/{freeze_path}/best_{freeze_path}_fold_longitudinal_estimator_params_CVAE2D_{args.dimension}_{args.beta}_{args.gamma}_{args.iterations}.json"


    # Saving the best VAE model in the right folder
    best_fold = CV_VAE(CVAE2D_ORIGINAL, [i for i in range(8)], test_set, nn_saving_path, dataset_name,
                        latent_dimension=latent_representation_size, gamma=gamma, beta=beta, batch_size=batch_size, freeze=freeze_path)
    print("Best VAE fold =", best_fold)
    
    best_fold_model = CVAE2D_ORIGINAL(latent_representation_size)
    best_fold_model.gamma = gamma
    best_fold_model.beta = beta
    best_fold_model.load_state_dict(torch.load(nn_saving_path+f"_fold_{best_fold}.pth", map_location='cpu'))
    torch.save(best_fold_model.state_dict(), save_best_fold_path_VAE)



    # Saving the best LVAE model in the right folder
    best_fold = CV_LVAE(CVAE2D_ORIGINAL, [i for i in range(8)], test_set, nn_saving_path, longitudinal_saving_path, dataset_name,
                        latent_dimension=latent_representation_size, gamma=gamma, beta=beta, freeze=freeze_path)
    print("Best LVAE fold =", best_fold)

    best_fold_model = CVAE2D_ORIGINAL(latent_representation_size)
    best_fold_model.gamma = gamma
    best_fold_model.beta = beta
    best_fold_model.load_state_dict(torch.load(nn_saving_path+f"_fold_{best_fold}.pth2", map_location='cpu'))
    torch.save(best_fold_model.state_dict(), save_best_fold_path_VAE+"2")
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path+f"_fold_{best_fold}.json2")
    longitudinal_estimator.save(save_best_fold_path_LVAE+"2")




    