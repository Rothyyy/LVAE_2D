#!/bin/bash

rm -f data_starmen/anomaly_images/*
rm -f data_starmen/anomaly_patches:*

python -m anomaly.generate_anomaly -n 5 -a growing_circle -p True
python -m anomaly.generate_anomaly -n 5 -a darker_line -p True
