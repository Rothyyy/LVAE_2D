# LVAE_2D

In this repository you will find the files from the longitudinal_anomaly_detection repository that will be useful for the synthetic starmen dataset. (I removed everything related to the 3D version for brain RMI)

Link to starmen dataset : https://zenodo.org/records/5081988 

You will need to download the dataset from the link above. From the `.gz` file obtained, go in `starmen/output_random` and extract the `images` folder and  `path_to_visit_ages_file.txt` file in the `data_starmen` folder.

To launch the code you can :

- **Launch**: `python dataset/dataset_to_csv.py` to get the `starmen_dataset.csv` file that will be used in the training. (Make sure that the images folder and the text file containing the ages are in the `data_starmen` folder)
- **Launch**: `python train_model_2D.py` to launch a training of the LVAE on the starmen dataset and get the plots and figure for starman 9.
- **Launch**: `python display_results.py` to load the trained model and get figures for 10 randomly chosen starmen.

The repository already contains the `starmen_dataset.csv` file and checkpoints for the models. After downloading the images and putting them in the right folder, you can launch step 3 and see the results in `results_reconstruction`.


# Content of the repository:

- The `data_starmen` folder contains the images and ages.
- The `dataset` folder contains a program to get the csv file using the data that will be used for training and testing. It also contains the `Dataset` class that will be used for the `DataLoader`.
- The `nnModels` folder contains the Neural Network models (the VAE), some utility functions, loss functions and training code.
- The `longitudinalModel` folder contains codes that will train the Leaspy estimator with the VAE. It contains utility functions and training and test codes.
- The `saved_models_2D` folder contains checkpoints for our models.
- The `utils_display` folder contains functions used to display generated images and save them in a pdf file.
- The `outputs_reconstruction` folder contains pdf file with loss plots during training with `train_model_2D.py`, it will also contains the generated images for subject 9.
- The `results_reconstruction` folder contains pdf file with generated images for randomly chosen subject with `display_result.py`.
