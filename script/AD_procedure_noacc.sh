#!/bin/bash

bash ./script/noacc_switch.sh
bash ./script/generate_anomaly.sh
python -m train.cross_validation
bash ./script/launch_compute_thresholds_noacc.sh
bash ./script/detect_anomaly_noacc.sh