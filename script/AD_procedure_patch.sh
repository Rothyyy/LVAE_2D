#!/bin/bash

bash ./script/generate_anomaly.sh
# python -m train.cross_validation
bash ./script/launch_compute_thresholds_patch.sh
bash ./script/detect_anomaly_patch.sh