#!/bin/bash

python -m anomaly.detect_anomaly_patch --method image -a growing_circle --dim 64 --beta 0.5
python -m anomaly.detect_anomaly_patch --method image -a darker_line --dim 64 --beta 0.5

python -m anomaly.detect_anomaly_patch --method image -a growing_circle --dim 32 --beta 0.5
python -m anomaly.detect_anomaly_patch --method image -a darker_line --dim 32 --beta 0.5
