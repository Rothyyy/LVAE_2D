#!/bin/bash                                                                     
                                     
#OAR -p cluster='kinovis' 
#OAR -q besteffort 
#OAR -l walltime=20    

cd /home/rsochet/Code/LVAE_2D

module load conda
conda activate env_LVAE_2D

python train_model_2D_KF.py --dataset noacc -f y
python train_model_2D_KF.py --dataset noacc -f n -skip y
python train_model_2D_KF.py --dataset noacc -f yy -skip y

