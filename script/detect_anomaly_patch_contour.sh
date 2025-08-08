#!/bin/bash

rm -r plots/fig_anomaly_reconstruction/darker_line
rm -r plots/fig_anomaly_reconstruction/growing_circle
# rm -r plots/fig_anomaly_reconstruction/darker_circle


python -m anomaly.detect_anomaly_patch_contour --method image -a growing_circle --dim 32 --beta 1.0
python -m anomaly.detect_anomaly_patch_contour --method image -a darker_line --dim 32 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_circle --dim 32 --beta 1.0

python -m anomaly.detect_anomaly_patch_contour --method image -a growing_circle --dim 64 --beta 1.0
python -m anomaly.detect_anomaly_patch_contour --method image -a darker_line --dim 64 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_circle --dim 64 --beta 1.0

# python -m anomaly.detect_anomaly_patch_contour --method image -a growing_circle --dim 16 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_line --dim 16 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_circle --dim 16 --beta 1.0

# python -m anomaly.detect_anomaly_patch_contour --method image -a growing_circle --dim 464 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_line --dim 464 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_circle --dim 464 --beta 1.0

# python -m anomaly.detect_anomaly_patch_contour --method image -a growing_circle --dim 332 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_line --dim 332 --beta 1.0
# python -m anomaly.detect_anomaly_patch_contour --method image -a darker_circle --dim 332 --beta 1.0


