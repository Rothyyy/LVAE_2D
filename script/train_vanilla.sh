#!/bin/bash                                                                     
                                     
#OAR -p cluster='kinovis' 
#OAR -q besteffort 
#OAR -l walltime=20                

module load conda
conda activate env_LVAE

cd /home/rsochet/Code/LVAE_2D

bash script_switch_to_noacc.sh
python train_model_2D.py --dataset noacc -f y
python train_model_2D.py --dataset noacc -f n

bash script_switch_to_acc.sh
python train_model_2D.py --dataset acc -f y
python train_model_2D.py --dataset acc -f n