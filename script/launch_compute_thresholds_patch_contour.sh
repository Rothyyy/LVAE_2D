#!/bin/bash


python -m thresholds.compute_threshold_patch --dim 16 --beta 1.0 --plot True
python -m thresholds.compute_threshold_patch --dim 32 --beta 1.0  --plot True
python -m thresholds.compute_threshold_patch --dim 332 --beta 1.0  --plot True
python -m thresholds.compute_threshold_patch --dim 364 --beta 1.0  --plot True

