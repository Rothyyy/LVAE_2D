#!/bin/bash

cd data_starmen
rm path_to_visit_ages_file.txt
rm -r images
unzip starmen_acc.zip
cd ..
python dataset/dataset_to_csv.py
python dataset/group_based_train_test_split.py