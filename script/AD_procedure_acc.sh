#!/bin/bash

bash ./script/acc_switch.sh
bash ./script/generate_anomaly.sh
python cross_validation.py --dataset acc -f y
python cross_validation.py --dataset acc -f yy
python cross_validation.py --dataset acc -f n
bash ./script/launch_compute_thresholds_acc.sh
bash ./script/detect_anomaly_acc.sh