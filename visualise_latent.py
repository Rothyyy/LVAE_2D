import numpy as np
import pandas as pd
import sklearn.manifold
import torch
from leaspy import AlgorithmSettings, Leaspy
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import os
import json
import matplotlib.pyplot as plt
import sklearn

from dataset.Dataset2D import Dataset2D
from longitudinalModel.fit_longitudinal_estimator_on_nn import fit_longitudinal_estimator_on_nn
from dataset.group_based_train_test_split import group_based_train_test_split

from nnModels.CVAE2D_ORIGINAL import CVAE2D_ORIGINAL
from nnModels.losses import image_reconstruction_error, pixel_reconstruction_error

from utils_display.display_individual_observations_2D import project_encodings_for_results
from dataset.LongitudinalDataset2D import LongitudinalDataset2D, longitudinal_collate_2D

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
    return encodings, logvars, projected_images
    # return torch.from_numpy(predicted_latent_variables[str(subject_id)]), logvars, projected_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-a", "--anomaly", type=str, required=False, default="original")
    parser.add_argument("--beta", type=float, required=False, default=5)
    parser.add_argument("-kf", type=str, required=False, default="y")
    parser.add_argument('-f', '--freeze', type=str, required=True, default='y',
                        help='freeze convolution layer ? default = y')   
    parser.add_argument('--dataset', type=str, required=True, default="noacc",
                        help='Use the models trained on dataset "acc" or "noacc"') 
    args = parser.parse_args()

    if args.freeze == "y":
        freeze_path = "freeze_conv"
    elif args.freeze == "yy":
        freeze_path = "freeze_all"
    else:
        freeze_path = "no_freeze"

    if args.kf == "y" or args.kf == "yy":
        LVAE_model_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/best_{freeze_path}_fold_CVAE2D_4_5_100_200.pth2"
        LVAE_estimator_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/best_{freeze_path}_fold_longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"
    else:
        LVAE_model_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/CVAE2D_4_5_100_200.pth2"
        LVAE_estimator_path = f"saved_models_2D/dataset_{args.dataset}/{freeze_path}/longitudinal_estimator_params_CVAE2D_4_5_100_200.json2"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_csv_path = "data_csv/starmen_test_set.csv"
    dataset = LongitudinalDataset2D(data_csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=longitudinal_collate_2D, num_workers=round(os.cpu_count()/4))

    model = CVAE2D_ORIGINAL(4)
    model.gamma = 100
    model.beta = 5
    model.load_state_dict(torch.load(LVAE_model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    model.training = False

    longitudinal_estimator = Leaspy.load(LVAE_estimator_path)

    encodings_list = []

    for data in dataloader:

        images = data[0].to(device)
        subject_id = data[2][0]
        mus_LVAE, logvars_LVAE, _, _ = model(images)
        for i in range(10):
            encodings_list.append(mus_LVAE[i].detach())


    encodings_array = np.array(encodings_list)
    print(encodings_array.shape)
    
    t_sne_model = sklearn.manifold.TSNE(2)
    encodings_transformed = t_sne_model.fit_transform(encodings_array)
    with open("tsne_result.json", 'w') as f:
        json.dump(encodings_transformed.tolist(), f, ensure_ascii=False)
    # plt.plot(encodings_transformed)
    # plt.show()





