#!/bin/bash

bash ./script/generate_anomaly.sh
bash ./script/launch_compute_thresholds.sh
bash ./script/detect_anomaly.sh