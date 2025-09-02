#!/bin/bash                                                                     
                                     
#OAR -p cluster='kinovis' 
#OAR -q besteffort 
#OAR -l walltime=48    

cd /home/rsochet/Code/LVAE_2D

module load conda
conda activate env_LVAE

python -m train.train_model_2D_KF --beta 1 --lpips 1 #--dimension 4 --beta 5 --gamma 100 --iterations 200

