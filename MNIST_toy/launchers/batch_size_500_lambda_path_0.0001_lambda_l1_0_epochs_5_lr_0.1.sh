#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=batch_size:500,lambda_path:0.0001,lambda_l1:0,epochs:5,lr:0.1-%N-%j.out
source ~/ENVS/pytorch/bin/activate
python ../main.py --lr=0.1 --batch-size=500 --epochs=5 --p=2 --lambda-pr=.0.0001 --lambda-l1=0
