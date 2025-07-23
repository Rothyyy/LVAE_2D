import pandas as pd
import torch
from leaspy import Data


def fit_longitudinal_estimator_on_nn(data_loader, model, device, longitudinal_estimator,
                                     longitudinal_estimator_settings):
    with torch.no_grad():
        encodings = []
        times = []
        ids = []
        for data in data_loader:
            encodings.append(model.encoder(data[0].float().to(device))[0]
                             .detach().clone().to(device)
                            )
            for i in range(len(data[1])):
                times.extend(data[1][i])
                ids.extend([data[2][i]] * len(data[1][i]))
        # print(encodings[0].size(),encodings[0], encodings[1])

        # WARNING: If in the sample there are crops at different position for a same individual then
        # the algorithm is not capable of being trained on it
        encodings = torch.cat(encodings)
        encodings_df = pd.DataFrame({'ID': ids, 'TIME': times})
        for i in range(encodings.shape[1]):
            encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                                encodings[:, i].detach().clone().tolist())
        encodings_df['ID'] = encodings_df['ID'].astype(str)
        # print(encodings_df.head())
        # encodings_df.to_csv("encodings.csv",
        #                     index_label=False)  # TODO: think about whether it's really useful

    try:
        encodings_data = Data.from_dataframe(encodings_df)
        longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
    except:
        print(encodings_df.columns)
        try:
            encodings_df = encodings_df.reset_index(drop=True)
            encodings_data = Data.from_dataframe(encodings_df)
            longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
        except:
            print()
            print("Error in dimension features")
            print("Model features:", longitudinal_estimator.model.features)
            print("Data features:", encodings_data.headers)
    return longitudinal_estimator, encodings_df


def fit_longitudinal_estimator_on_nn_patch(data_loader, model, device, longitudinal_estimator,
                                           longitudinal_estimator_settings, patch_size=15):
    num_patch_per_image = 2500  # (64-patch_size//2)*(64-patch_size//2) = 50*50 = 2500
    with torch.no_grad():
        encodings = []
        times = []
        patch_ids = []

        for data in data_loader:
            encodings.append(model.encoder(data[0].float().to(device))[0]
                             .detach().clone().to(device)
                            )
            for i in range(len(data[1])):
                for image_timestamp in range(10):
                    times.extend([data[1][i][image_timestamp]] * num_patch_per_image)
                    patch_ids.extend(data[3][i][image_timestamp])

        # WARNING: If in the sample there are crops at different position for a same individual then
        # the algorithm is not capable of being trained on it
        encodings = torch.cat(encodings)
        encodings_df = pd.DataFrame({'ID': patch_ids, 'TIME': times})
        for i in range(encodings.shape[1]):
            encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                                encodings[:, i].detach().clone().tolist())
        encodings_df['ID'] = encodings_df['ID'].astype(str)

    try:
        encodings_data = Data.from_dataframe(encodings_df)
        longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
    except:
        print(encodings_df.columns)
        try:
            encodings_df = encodings_df.reset_index(drop=True)
            encodings_data = Data.from_dataframe(encodings_df)
            longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
        except:
            print()
            print("Error in dimension features")
            print("Model features:", longitudinal_estimator.model.features)
            print("Data features:", encodings_data.headers)
    return longitudinal_estimator, encodings_df

