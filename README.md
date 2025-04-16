# LVAE_2D

In this repository you will find the files from the longitudinal_anomaly_detection repository that will be useful for the synthetic starmen dataset. (I removed everything related to the 3D version for brain RMI)

Link to starmen dataset : https://zenodo.org/records/5081988 

You will need to download the dataset from the link above. From the `.gz` file obtained, go in `starmen/output_random` and extract the `images` folder and  `path_to_visit_ages_file.txt` file in the `data_starmen` folder.

To launch the code you can :

- **Launch**: `python dataset/dataset_to_csv.py` to get the `starmen_dataset.csv` file that will be used in the training. (Make sure that the images folder and the text file containing the ages are in the `data_starmen` folder)
- **Launch**: `python train_model_2D.py` to launch a training of the LVAE on the starmen dataset and get the plots and figure for starman 9.
- **Launch**: `python display_results.py` to load the trained model and get figures for 10 randomly chosen starmen.

The repository already contains the `starmen_dataset.csv` file and checkpoints for the models. After downloading the images and putting them in the right folder, you can launch step 3 and see the results in `results_reconstruction`.