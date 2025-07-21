#!/bin/bash

cd data_starmen
rm path_to_visit_ages_file.txt
rm -r images
unzip starmen_noacc.zip
cd ..
python -m dataset.dataset_to_csv
python -m dataset.group_based_train_test_split