# LVAE_2D


Link to starmen dataset : https://zenodo.org/records/5081988
You will need to download the dataset from the link above. From the `.gz` file obtained, go in `starmen/output_random` and extract the `images` folder and  `path_to_visit_ages_file.txt` file in the data_starmen folder.

Launch: `python dataset/dataset_to_csv.py` to get the starmen_dataset.csv file that will be used in the training.
Launch: `python train_model_2D.py` to launch a training of the LVAE on the starmen dataset and get the plots and figure for a random starman.
Launch: `python display_results.py` to load the trained model and get figures for 10 randomly chosen starmen.
