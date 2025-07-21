#!/bin/bash                                                                     
                                     
#OAR -p cluster='kinovis' 
#OAR -q besteffort 
#OAR -l walltime=20    

cd /home/rsochet/Code/LVAE_2D

module load conda
conda activate env_LVAE_2D

python -m train_model_2D_KF --dimension 4 --beta 5 --gamma 100 --iterations 200

