#!/bin/bash                                                                     
                                     
#OAR -p cluster='kinovis' 
#OAR -q besteffort 
#OAR -l walltime=48    

cd /home/rsochet/Code/LVAE_2D

module load conda
conda activate env_LVAE

python -m train.train_model_2D_KF_patch --dimension 32 --beta 0.5 --gamma 5 --iterations 20 -skip y

