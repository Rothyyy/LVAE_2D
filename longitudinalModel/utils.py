import torch
import pandas as pd


def produce_encodings_df(model, data, device):
    encodings = []
    times = []
    ids = []
    encodings.append(model.encoder(data[0].float().to(device))[
                         0].detach().clone())

    for i in range(len(data[1])):
        times.extend(data[1][i])
        ids.extend([data[2][i]] * len(data[1][i]))
    encodings = torch.cat(encodings)
    encodings_df = pd.DataFrame({'ID': ids, 'TIME': times})
    for i in range(encodings.shape[1]):
        encodings_df.insert(len(encodings_df.columns), f"ENCODING{i}",
                            encodings[:, i].detach().clone().tolist())
    encodings_df['ID'] = encodings_df['ID'].astype(str)
    return encodings_df


def produce_encodings_patch_contour_df(model, data, device):
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
        encodings_df['ID'] = encodings_df['ID'].astype(str)