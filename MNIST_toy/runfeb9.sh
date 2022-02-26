#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=adam.weiss@mail.mcgill.ca
#SBATCH --mail-type=ALL

#./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) test.csv test 0
#./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) results_cifar10_resnet.csv cifar10_resnet 0
#./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) marabou_benchmarks.csv marabou-cifar10 0
#./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) mnistfc.csv mnistfc 0
python feb9.py