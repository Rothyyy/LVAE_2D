#!/bin/bash

# python compute_threshold.py -m pixel_all -kf y -f y -set test --dataset acc
# python compute_threshold.py -m pixel_all -kf y -f yy -set test --dataset acc
# python compute_threshold.py -m pixel_all -kf y -f n -set test --dataset acc


# python compute_threshold.py -m pixel -kf y -f y -set test --dataset acc
# python compute_threshold.py -m pixel -kf y -f yy -set test --dataset acc
python compute_threshold.py -m pixel -kf y -f n -set test --dataset acc


# python compute_threshold.py -m image -kf y -f y -set test --dataset acc
# python compute_threshold.py -m image -kf y -f yy -set test --dataset acc
python compute_threshold.py -m image -kf y -f n -set test --dataset acc