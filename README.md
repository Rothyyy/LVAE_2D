# LVAE_2D

In this repository you will find the files from the longitudinal_anomaly_detection repository that will be useful for the synthetic starmen dataset.

Link to starmen dataset : https://zenodo.org/records/5081988 

This github repository should have in folder `data_starmen` a zip file named `starmen_noacc.zip`, this file contains the images in .npy format and a text file containing the times of visits for each starmen. 


# Requirements and script to launch training and anomaly detection processes:

Before launching any script, make sure to use Python version 3.11 (for Leaspy package) and install all the necessary package with `requirements.txt`, preferably in a virtual environment.


The folder `script` contains multiple `.sh` file that will launch some parts of the code, before launching any of them, make sure to first execute the file: `noacc_switch.sh` this script will unzip the zip file containing the images of the dataset, it will also split and create `.csv` files of the train/test/validation which will be used by the dataloader.

If you intend to use the patch version you should also launch `get_patches_contour.sh`

Summary:
- Use Python 3.11, install required packages with `requirements.txt` (for example `pip install -r requirements.txt`) 
- **Launch** script to get the image dataset: `bash ./script/noacc_switch.sh`
- (Optional) **Launch** script to get the patches dataset from images: `bash ./script/get_patches_contour.sh`



## Launch training

Before launching the training, make sure to have the dataset properly extracted with the script presented above.

To launch the training for VAE and LVAE with whole images execute: `python -m train.train_model_2D_KF` \
To launch the training for VAE with patch version execute: `python -m train.train_model_2D_KF_patch`


The repository should already contain the checkpoints for the models. After extracting the images (and patches) with previous scripts, you can skip this section and start the process for unsupervised anomaly detection.


## Launch unsupervised anomaly detection (UAD)

The UAD pipeline is done in three steps, first we compute the thresholds, second we generate anomalous images, third we perform UAD using the model's checkpoints, thresholds and anomalous images. Each of these steps can be launched individually, but for simplicity the following script will execute the whole pipeline:

- For whole images: `bash ./script/AD_procedure.sh`
- For patches: `bash ./script/AD_procedure_patch_contour.sh`

These scripts will launch and execute the following steps:


The first step in the UAD pipeline is to compute the thresholds, this can be done by launching the following script:

- For whole images: `bash ./script/launch_compute_thresholds.sh`
- For patches: `bash ./script/launch_compute_thresholds_patch_contour.sh`

These script will do an epoch on the test set to get the reconstruction error when considering the image and pixel (for the first script), when considering the patches (for the second script).

The next step is to generate anomalies, to do that launch: `bash ./script/generate_anomaly.sh`. This will generate and create five anomalous images and extract the patches for each anomaly.

The last step is to perform UAD by launching these scripts:

- For whole images: `bash ./script/detect_anomaly.sh`
- For patches: `bash ./script/detect_anomaly_patch_contour.sh`

The program will load the checkpoints, thresholds and perform UAD on anomalous images. The results will be saved in `pdf` files in `./plots/fig_anomaly_reconstruction`.




# Content of the repository:

- The `data_starmen` folder contains the images and ages.
- The `dataset` folder contains a program to get the csv file using the data that will be used for training and testing. It also contains the `Dataset` class that will be used for the `DataLoader`.
- The `data_csv` folder will contain csv files for the different dataset, training, validation, testing, anomalous, as well as the treshold csv file used for anomaly detection.
- The `nnModels` folder contains the Neural Network models (the VAE), some utility functions, loss functions and training code.
- The `longitudinalModel` folder contains codes that will train the Leaspy estimator with the VAE. It contains utility functions and training and test codes.
- The `saved_models_2D` folder contains checkpoints for our models.
- The `utils` folder contains functions used to display generated images and save them in a pdf file.
- The `anomaly` folder contains files to generate some anomaly on the starmen dataset and perform anomaly detection. In `figure_reconstruction` you will find  results of the anomaly detection done on these figures. The bar plots shows at which timestamp the model detects an anomaly (this is done only with the `image` method). The figures, divided by the type of anomaly, shows three rows of starmen images, the first is the original image containing the anomaly, the second row shows the generated image by the VAE or LVAE, the third row shows a residual depending on the method used.


# More details on the anomaly detection workflow

- First, we generate the anomalous dataset, currently we can add four different kinds of anomaly : [`growing_circle`, `darker_circle`, `darker_line`, `shrinking_circle`], these anomaly consist in adding a modification on the starmen's left arm that grows more (or less) apparent with time.

    To do that, launch : `python -m anomaly.generate_anomaly -n num_sample -a anomaly_type -pc True` where `num_sample` is the number of starmen to select (5 by default), `anomaly_type` is one of the four anomaly described aboven `-pc True` will also extract patches. 
    As output, you will obtain the anomalous images in npy format stored in `data_starmen/anomaly_images` and `data_starmen/anomaly_patches` folder and the corresponding csv file in `data_csv` folder. You will also obtain plots of histogram from the reconstruction loss computed. 


- The second step is to compute and store the different threshold that will be used during the detection.

    To do that, launch : `python -m thresholds.compute_threshold --method method_choice` where `method_choice` is one of the following : [`image`, `pixel`, `pixel_all`]. Depending on the method used the threshold's computation will be different. As output you will obtain a csv file in folder `data_csv` containing some statistics and the tresholds. For the patch version launch : `python -m thresholds.compute_threshold_patch --dim 32 --beta 1.0`, only `image` method is implemented for the patch version.


- The last step is to launch the anomaly detection.
    
    To do that launch : `python -m anomaly.detect_anomaly --method method_choice -a anomaly_type` where `anomaly_type` is one of the following [`growing_circle`, `darker_circle`, `darker_line`, `shrinking_circle`] and `method_choice` [`image`, `pixel`, `pixel_all`]. For patch version, launch: `python -m anomaly.detect_anomaly_patch_contour -a anomaly_type --dim 32 --beta 1.0`

    As output you will obtain figures in the folder `anomaly/figure_reconstruction/anomaly_type/method` where you will find three rows with orginal image, reconstruction and residual.


