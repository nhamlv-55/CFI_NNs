#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=batch_size:10,lambda_path:0.1,lambda_l1:0.1,epochs:1,lr:0.1-%N-%j.out
source ~/ENVS/pytorch/bin/activate
python ../main.py --lr=0.1 --batch-size=10 --epochs=1 --p=2 --lambda-pr=.0.1 --lambda-l1=0.1