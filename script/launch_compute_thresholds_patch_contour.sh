#!/bin/bash


# python -m thresholds.compute_threshold_patch --dim 16 --beta 1.0
python -m thresholds.compute_threshold_patch --dim 32 --beta 1.0
python -m thresholds.compute_threshold_patch --dim 64 --beta 1.0

# python -m thresholds.compute_threshold_patch --dim 432 --beta 1.0
# python -m thresholds.compute_threshold_patch --dim 364 --beta 1.0

