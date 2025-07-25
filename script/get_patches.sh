#!/bin/bash

cd data_starmen
rm -r images_patch
cd ..
python ./dataset/patch_to_csv.py -t 22
