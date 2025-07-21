#!/bin/bash

python detect_anomaly.py --dataset noacc -m image -a growing_circle -f y
python detect_anomaly.py --dataset noacc -m image -a growing_circle -f yy
python detect_anomaly.py --dataset noacc -m image -a growing_circle -f n

python detect_anomaly.py --dataset noacc -m image -a darker_line -f y
python detect_anomaly.py --dataset noacc -m image -a darker_line -f yy
python detect_anomaly.py --dataset noacc -m image -a darker_line -f n


python detect_anomaly.py --dataset noacc -m pixel -a growing_circle -f y
python detect_anomaly.py --dataset noacc -m pixel -a growing_circle -f yy
python detect_anomaly.py --dataset noacc -m pixel -a growing_circle -f n

python detect_anomaly.py --dataset noacc -m pixel -a darker_line -f y
python detect_anomaly.py --dataset noacc -m pixel -a darker_line -f yy
python detect_anomaly.py --dataset noacc -m pixel -a darker_line -f n

python detect_anomaly.py --dataset noacc -m pixel_all -a growing_circle -f y 
python detect_anomaly.py --dataset noacc -m pixel_all -a growing_circle -f yy
python detect_anomaly.py --dataset noacc -m pixel_all -a growing_circle -f n

python detect_anomaly.py --dataset noacc -m pixel_all -a darker_line -f y
python detect_anomaly.py --dataset noacc -m pixel_all -a darker_line -f yy
python detect_anomaly.py --dataset noacc -m pixel_all -a darker_line -f n

