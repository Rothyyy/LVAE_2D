import pandas as pd
import torch
from leaspy import Data


def fit_longitudinal_estimator_on_nn_patch_contour(data_loader, model, device, longitudinal_estimator,
                                     longitudinal_estimator_settings):
    with torch.no_grad():
        encodings = []
        times = []
        patch_ids = []
        for data in data_loader:
            time_patch = data[1]
            subject_ids = data[2]
            centers_list = data[3]
            batch_size = len(centers_list)
            
            encodings_output = model.encoder(data[0].float().to(device))[0].detach().clone().to(device)
            encodings.append(encodings_output.view(encodings_output.size(0), encodings_output.size(1)))

            for i in range(batch_size):
                for t in range(10):
                    number_patches = len(centers_list[i][t])
                    times.extend([time_patch[i][t]] * number_patches)
                    for patch in range(number_patches):
                        center_coord_x, center_coord_y = centers_list[i][t][patch]
                        patch_id = f"s_{subject_ids[i]}_{int(center_coord_x)}_{int(center_coord_y)}"
                        patch_ids.append(patch_id)
                
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
            print("Error in fit")
    return longitudinal_estimator, encodings_df



def fit_longitudinal_estimator_on_nn_patch_contour_v1(data, model, device, longitudinal_estimator,
                                           longitudinal_estimator_settings):
    with torch.no_grad():
        encodings = []
        times = []
        patch_ids = []

        time_patch = data[1]
        subject_ids = data[2]
        centers_list = data[3]
        batch_size = len(centers_list)

        encodings_output = model.encoder(data[0].float().to(device))[0].detach().clone().to(device)
        encodings.append(encodings_output.view(encodings_output.size(0), encodings_output.size(1)))
        
        for i in range(batch_size):
            for t in range(10):
                number_patches = len(centers_list[i][t])
                times.extend([time_patch[i][t]] * number_patches)
                for patch in range(number_patches):
                    center_coord_x, center_coord_y = centers_list[i][t][patch]
                    patch_id = f"s_{subject_ids[i]}_{int(center_coord_x)}_{int(center_coord_y)}"
                    patch_ids.append(patch_id)

        # WARNING: If in the sample there are crops at different position for a same individual then
        # the algorithm is not capable of being trained on it
        encodings = torch.cat(encodings)
        encodings_df = pd.DataFrame({'ID': patch_ids, 'TIME': times})
        for i in range(encodings.shape[1]):
            encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                                encodings[:, i].detach().clone().tolist())
        # encodings_df['ID'] = encodings_df['ID'].astype(str)
    # encodings_df.to_csv("./encodings.csv")
    # exit()
    try:
        print("Try 1")
        encodings_data = Data.from_dataframe(encodings_df)
        longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
    except:
        print("Try 2")
        # try:
        encodings_df = encodings_df.reset_index(drop=True)
        encodings_data = Data.from_dataframe(encodings_df)
        longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
        # except:
        #     print()
        #     print("Error in dimension features")
        #     print("Model features:", longitudinal_estimator.model.features)
        #     print("Data features:", encodings_data.headers)
    return longitudinal_estimator, encodings_df, patch_ids




def fit_longitudinal_estimator_on_nn_patch_contour_v2(data_loader, model, device, longitudinal_estimator,
                                                      longitudinal_estimator_settings):
    with torch.no_grad():
        encodings = []
        times = []
        patch_ids = []

        for data in data_loader:
            time_patch = data[1]
            subject_ids = data[2]
            centers_list = data[3]
            batch_size = len(centers_list)
            
            encodings_output = model.encoder(data[0].float().to(device))[0].detach().clone().to(device)
            encodings.append(encodings_output.view(encodings_output.size(0), encodings_output.size(1)))

            for i in range(batch_size):
                for t in range(10):
                    number_patches = len(centers_list[i][t])
                    times.extend([time_patch[i][t]] * number_patches)
                    for patch in range(number_patches):
                        center_coord_x, center_coord_y = centers_list[i][t][patch]
                        patch_id = f"s_{subject_ids[i]}_{int(center_coord_x)}_{int(center_coord_y)}"
                        patch_ids.append(patch_id)

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
            print("Error in fit")
    return longitudinal_estimator, encodings_df, patch_ids



