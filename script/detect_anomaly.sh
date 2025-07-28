#!/bin/bash

python -m anomaly.detect_anomaly --method image -a growing_circle
python -m anomaly.detect_anomaly --method image -a darker_line

python -m anomaly.detect_anomaly --method pixel -a growing_circle
python -m anomaly.detect_anomaly --method pixel -a darker_line

# python detect_anomaly.py --dataset noacc -m pixel_all -a growing_circle -f y 
# python detect_anomaly.py --dataset noacc -m pixel_all -a growing_circle -f yy
# python detect_anomaly.py --dataset noacc -m pixel_all -a growing_circle -f n

# python detect_anomaly.py --dataset noacc -m pixel_all -a darker_line -f y
# python detect_anomaly.py --dataset noacc -m pixel_all -a darker_line -f yy
# python detect_anomaly.py --dataset noacc -m pixel_all -a darker_line -f n

