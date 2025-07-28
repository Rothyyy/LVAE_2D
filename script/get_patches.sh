#!/bin/bash

cd data_starmen
rm -r images_patch
cd ..
python -m dataset.patch_to_csv -t 50
python -m dataset.group_based_train_test_split -p True
python -m dataset.split_k_folds -p True