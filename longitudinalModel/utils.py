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
