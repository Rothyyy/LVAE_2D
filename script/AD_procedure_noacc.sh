#!/bin/bash

bash ./script/noacc_switch.sh
bash ./script/generate_anomaly.sh
python cross_validation.py --dataset noacc -f y
python cross_validation.py --dataset noacc -f yy
python cross_validation.py --dataset noacc -f n
bash ./script/launch_compute_thresholds_noacc.sh
bash ./script/detect_anomaly_noacc.sh