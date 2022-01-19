import os
import itertools

batch_size = [10,100,500,100] #hyperparamters
lpath = [0,.0001,.001,.01,.1,1]
epochs = [1,5,10]
l1 =[0,.1,1]
lr = [.01,.1]

base = '''#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output='''

next_line = "source ~/ENVS/pytorch/bin/activate\n"

for b,lp,l1,e,rate in itertools.product(batch_size, lpath, l1, epochs,lr):
    output_1 = "batch_size_{}_lambda_path_{}_lambda_l1_{}_epochs_{}_lr_{}.sh".format(b,lp,l1,e,rate)
    output_name = "batch_size:{},lambda_path:{},lambda_l1:{},epochs:{},lr:{}-%N-%j.out\n".format(b,lp,l1,e,rate)
    command = "python ../main.py --lr={} --batch-size={} --epochs={} --p=2 --lambda-pr=.{} --lambda-l1={}\n".format(rate,b,e,lp,l1)
    #print(output_name)
    #print(command)
    script = base + output_name + next_line + command
    print(script)
    f = open("launchers/"+output_1,"w")
    f.write(script)
    f.close()
    print(output_1)
    os.system("sbatch {}".format("launchers/"+output_1))
    
    
    #input()