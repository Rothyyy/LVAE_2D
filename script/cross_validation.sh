#!/bin/bash

python cross_validation --dataset acc -f y
python cross_validation --dataset acc -f yy
python cross_validation --dataset acc -f n

python cross_validation --dataset noacc -f y
python cross_validation --dataset noacc -f yy
python cross_validation --dataset noacc -f n