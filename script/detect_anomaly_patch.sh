#!/bin/bash

rm -r plots/fig_anomaly_reconstruction/darker_line
rm -r plots/fig_anomaly_reconstruction/growing_circle


python -m anomaly.detect_anomaly_patch --method image -a growing_circle --dim 64 --beta 0.5
python -m anomaly.detect_anomaly_patch --method image -a darker_line --dim 64 --beta 0.5

python -m anomaly.detect_anomaly_patch --method image -a growing_circle --dim 32 --beta 0.5
python -m anomaly.detect_anomaly_patch --method image -a darker_line --dim 32 --beta 0.5

# python -m anomaly.detect_anomaly_patch --method image -a growing_circle --dim 16 --beta 0.5
# python -m anomaly.detect_anomaly_patch --method image -a darker_line --dim 16 --beta 0.5
