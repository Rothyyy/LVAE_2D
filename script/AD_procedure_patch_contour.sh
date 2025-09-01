#!/bin/bash

bash ./script/generate_anomaly.sh
bash ./script/launch_compute_thresholds_patch_contour.sh
bash ./script/detect_anomaly_patch_contour.sh

