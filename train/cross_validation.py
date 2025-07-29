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

from dataset.Dataset2D import Dataset2D, Dataset2D_patch, collate_2D_patch
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D
from dataset.LongitudinalDataset2D_patch import LongitudinalDataset2D_patch, longitudinal_collate_2D_patch

from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from longitudinalModel.project_encodings_for_training import project_encodings_for_training

from utils.project_encodings_for_results import project_encodings_for_results
from utils.display_individual_observations_2D import get_longitudinal_images



def CV_VAE(model_type, fold_index_list, test_set, nn_saving_path,
           device='cuda' if torch.cuda.is_available() else 'cpu', plot_save_path=None,
           latent_dimension=4, beta=5, gamma=100,
           batch_size=256, num_worker=round(os.cpu_count()/4), cv_patch=False, patch_size=15):

    best_fold = 0
    best_loss = torch.inf
    folds_test_loss = np.zeros(len(fold_index_list))
    transformations = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32))
            , transforms.Lambda(lambda x: 2*x - 1)
        ])
    if cv_patch:
        dataset = Dataset2D_patch(test_set, transform=transformations)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True, pin_memory=True, collate_fn=collate_2D_patch)
    else:
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


        with torch.no_grad():
            for x in data_loader:
                if cv_patch:
                    x = x.reshape(-1, 1, patch_size, patch_size)
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

        test_mean_loss = sum(losses) / len(losses)
        folds_test_loss[fold_index] = test_mean_loss.detach().to("cpu")

        if test_mean_loss < best_loss:
            best_loss = test_mean_loss
            best_fold = fold_index

    if plot_save_path is None:
        plot_save_path = f"plots/training_plots/folds/CVAE2D_{latent_dimension}_{beta}/CV_VAE_results.pdf"
        os.makedirs(f"plots/training_plots/folds/CVAE2D_{latent_dimension}_{beta}", exist_ok=True)
    else:
        os.makedirs(plot_save_path, exist_ok=True)
        plot_save_path += f"CV_VAE_results.pdf"
    plt.plot(fold_index_list, folds_test_loss)
    plt.savefig(plot_save_path)
    plt.clf()

    return best_fold

def CV_LVAE(model_type, fold_index_list, test_set, nn_saving_path, longitudinal_saving_path,
           device='cuda' if torch.cuda.is_available() else 'cpu', plot_save_path=None,
           latent_dimension=4, gamma=100, beta=5, iterations=200,
           num_worker=round(os.cpu_count()/4), cv_patch=False):

    transformations = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32))
            , transforms.Lambda(lambda x: 2*x - 1)
        ])
    if cv_patch:
        dataset = LongitudinalDataset2D_patch(test_set, transform=transformations)
        test_data_loader = DataLoader(dataset, batch_size=1, num_workers=num_worker, shuffle=True, pin_memory=True, collate_fn=longitudinal_collate_2D_patch)
    else:
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

        with torch.no_grad():

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
        plot_save_path = f"plots/training_plots/LVAE_folds/CVAE2D_{latent_dimension}_{beta}_{gamma}_{iterations}/CV_LVAE_results.pdf"
    else:
        plot_save_path += f"CV_LVAE_results.pdf"
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
    parser.add_argument('--skip', type=bool, required=False, default=False)
    args = parser.parse_args()

    test_set_path = "./data_csv/starmen_test_set.csv"
    test_set = pd.read_csv(test_set_path)

    batch_size = args.batch_size
    latent_representation_size = args.dimension
    gamma = args.gamma
    beta = args.beta
    iterations = args.iterations

    # Path to saved model
    VAE_saving_path = f'saved_models_2D/VAE_folds/CVAE2D_{latent_representation_size}_{beta}'
    LVAE_saving_path = f'saved_models_2D/LVAE_folds/CVAE2D_{latent_representation_size}_{beta}_{gamma}_{iterations}'
    longitudinal_saving_path = f'saved_models_2D/LVAE_folds/longitudinal_estimator_params_CVAE2D_{latent_representation_size}_{beta}_{gamma}_{iterations}'

    # Path to save the best fold
    save_best_fold_path_VAE = f"saved_models_2D/best_fold_CVAE2D_{latent_representation_size}_{beta}.pth"
    save_best_fold_path_LVAE = f"saved_models_2D/best_fold_CVAE2D_{latent_representation_size}_{beta}_{gamma}_{iterations}.pth"
    save_best_fold_path_longitudinal_estimator = f"saved_models_2D/best_fold_longitudinal_estimator_params_CVAE2D_{latent_representation_size}_{beta}_{gamma}_{args.iterations}.json"


    # Saving the best VAE model in the right folder
    if args.skip == False:
        best_fold = CV_VAE(CVAE2D_ORIGINAL, [i for i in range(8)], test_set, VAE_saving_path,
                            latent_dimension=latent_representation_size, gamma=gamma, beta=beta, batch_size=batch_size)
        print("Best VAE fold =", best_fold)
        
        best_fold_model = CVAE2D_ORIGINAL(latent_representation_size)
        best_fold_model.gamma = gamma
        best_fold_model.beta = beta
        best_fold_model.load_state_dict(torch.load(VAE_saving_path+f"_fold_{best_fold}.pth", map_location='cpu'))
        torch.save(best_fold_model.state_dict(), save_best_fold_path_VAE)



    # Saving the best LVAE model in the right folder
    best_fold = CV_LVAE(CVAE2D_ORIGINAL, [i for i in range(8)], test_set, LVAE_saving_path, longitudinal_saving_path,
                        latent_dimension=latent_representation_size, gamma=gamma, beta=beta)
    print("Best LVAE fold =", best_fold)

    best_fold_model = CVAE2D_ORIGINAL(latent_representation_size)
    best_fold_model.gamma = gamma
    best_fold_model.beta = beta
    best_fold_model.load_state_dict(torch.load(LVAE_saving_path+f"_fold_{best_fold}.pth2", map_location='cpu'))
    torch.save(best_fold_model.state_dict(), save_best_fold_path_LVAE+"2")
    longitudinal_estimator = Leaspy.load(longitudinal_saving_path+f"_fold_{best_fold}.json2")
    longitudinal_estimator.save(save_best_fold_path_longitudinal_estimator+"2")




    