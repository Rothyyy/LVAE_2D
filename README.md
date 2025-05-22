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
- The `data_csv` folder will contain csv files for the different dataset, training, validation, testing, anomalous, as well as the treshold csv file used for anomaly detection.
- The `nnModels` folder contains the Neural Network models (the VAE), some utility functions, loss functions and training code.
- The `longitudinalModel` folder contains codes that will train the Leaspy estimator with the VAE. It contains utility functions and training and test codes.
- The `saved_models_2D` folder contains checkpoints for our models.
- The `utils_display` folder contains functions used to display generated images and save them in a pdf file.
- The `outputs_reconstruction` folder contains pdf file with loss plots during training with `train_model_2D.py`, it will also contains the generated images for subject 9.
- The `results_reconstruction` folder contains pdf file with generated images for randomly chosen subject with `display_result.py`.
- The `anomaly` folder contains a file to generate some anomaly on the starmen dataset. In `figure_reconstruction` you will find  results of the anomaly detection done on these figures. The bar plots shows at which timestamp the model detects an anomaly (this is done only with the `image` method). The figures, divided by the type of anomaly, shows three rows of starmen images, the first is the original image containing the anomaly, the second row shows the generated image by the VAE or LVAE, the third row shows a residual depending on the method used.


# The anomaly detection workflow

- First, we generate the anomalous dataset, currently we can add three different kind of anomaly : [`growing_circle`, `darker_circle`, `darker_line`, `shrinking_circle`], these anomaly consist in adding a modification that grows more apparent with time.

    To do that, launch : `python anomaly/generate_anomaly.py -n num_sample -a anomaly_type` where `num_sample` is the number of starmen to select (10 by default), and `anomaly_type` is one of the three anomaly described above. As output, you will obtain the anomalous images in npy format stored in `data_starmen/anomaly_images` folder and the corresponding csv file in `data_csv` folder. You will also obtain plots of histogram from the reconstruction loss computed.


- The second step is to compute and store the different threshold that will be used during the detection.

    To do that, launch : `python compute_threshold.py -m method` where method is one of the following : [`image`, `pixel`, `pixel_all`]. Depending on the method used the threshold's computation will be different. As output you will obtain a csv file in folder `data_csv` containing some statistics and the tresholds.


- The last step is to launch the anomaly detection.
    
    To do that launch : `python detect_anomaly_.py -a anomaly_type -m method` where anomaly type is one of the following [`growing_circle`, `darker_circle`, `darker_line`, `shrinking_circle`] and method [`image`, `pixel`, `pixel_all`].

    As output you will obtain figures in the folder `anomaly/figure_reconstruction/anomaly_type/method` where you will find three rows with orginal image, reconstruction and residual.


