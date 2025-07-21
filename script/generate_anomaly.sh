#!/bin/bash

rm -f data_starmen/anomaly_images/*

python anomaly/generate_anomaly.py -n 5 -a growing_circle
python anomaly/generate_anomaly.py -n 5 -a darker_line 
# python anomaly/generate_anomaly.py -n 5 -a shrinking_circle  
# python anomaly/generate_anomaly.py -n 5 -a darker_circle 
