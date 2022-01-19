#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=E5-ADA-GPU-SPEED-TEST-100-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=adam.weiss@mail.mcgill.ca
#module load cuda cudnn
source ~/ENVS/pytorch/bin/activate
python main.py --lr=0.1 --batch-size=100 --epochs=10 --p=2 --lambda-pr=.01
#python -m torch.utils.bottleneck main.py --lr=0.1 --batch-size=10 --epochs=1 --p=2 --lambda-pr=.01