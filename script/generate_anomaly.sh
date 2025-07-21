#!/bin/bash

rm -f data_starmen/anomaly_images/*

python -m anomaly.generate_anomaly -n 5 -a growing_circle -p False
python -m anomaly.generate_anomaly -n 5 -a darker_line -p False
