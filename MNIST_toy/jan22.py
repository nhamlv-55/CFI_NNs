# %%
from audioop import bias
from models import FeedforwardNeuralNetModel, TinyCNN, PatternClassifier
import regularizer_losts as rl
from torchvision import datasets, transforms
from torch import optim
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
import utils as CFI_utils
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict


import seaborn as sns
colors = sns.color_palette("tab10")

#Logging stuffs
import logging
import sys
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set STDERR handler as the only handler 
logger.handlers = [handler]


#configs
epochs = 10
batch_size = 1000
test_batch_size = 10000
use_cuda = True
lr = 0.01
log_interval = 100


#torch specific configs
torch.manual_seed(1)

#device = torch.device("cuda")
device = torch.device("cpu")
train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# %% [markdown]
# ## Agenda
# ### Big question: CFI for NN
# 
# 
# Are we doing complete verification or fuzzing/testing?
# 
# What is a `CFI`?
# 
# # What is a good abstraction for a NN's structure? Can it be enforced through regularization?
# - Distillation? (https://arxiv.org/pdf/1503.02531.pdf)
# - Decision Tree? (https://arxiv.org/pdf/1711.09784.pdf)
# - Model Extraction? (https://arxiv.org/pdf/2003.04884.pdf)
# 
# 
# ### Where are we?
# We want to do CFI for NN
# 
# ## Question: Do we want to verify each AP separately? Or we want to tell the difference between activation patterns of real images from fake one using a method M and verify M?
# 
# 
# 
# ## Option 1: We want to verify each AP separately (we are here)
# To be able to do so, we need
# 
# ### Condition 1: The number of activation patterns for a neural networks has to be way smaller than the size of the training/test set.
# #### Why: 
# epoch
#   If every input results in a different AP, given an AP, very likely during infer time every input will have its own AP, so we have to reject it
# 
#   We want to verify each AP separately, so smaller the number the better
# 
# #### Progress: 
# 
#   with the current regularization effort, we have been able to reduce the number of AP to about 20k for 60k input.
# 
# #### Aim: 
# 
# Reduce to about few hundreds
# 
# #### Tricky part: 
# 
# Given a network of N layers, we can limit the number of activation patterns by only looking at k < N layers. Question: which k? 
# 
# ### Condition 2: All, or most, of the activation patterns of the test set has to be covered by the AP of the training set
# #### Why: 
# If the activation patterns in the test set are completely different from the training set, there would be too many false negative at the inferencing time
# #### Progress: 
# Possible, but with a big caveat: the fake AP is now also included in the set
# 
# ### Condition 3: Given an adv exp, its AP cannot be in the training AP set.
# #### Progress:
# Currently, if Cond.2 is satisfied, then 3 is not.
# 
# 
# ## Option 2: Train the network s.t the fake AP and real AP can be separated by some methods (linear classifier or a small neural net)
# 
# ## What are the state-of-the-art attacks and defences?
# 
# 
# ## Backlog RQs: 
# - BRQ1: Can we look at a random k entries in the weight instead of the first k?
# - BRQ2: What if we set all the small abs weight in the Pattern classifer to 0? what is the accuracy in that case?
# - BRQ4: Given a simple attacking method (e.g, Fast gradient sign), build a dataset of activation maps of true and fake digits. Try to train a pattern classifier using that dataset.
# - __BRQ5: Given a target and its corresponding adv. exp, can we gradually move in the in-between space to see at which point the label is changed?__
# 
# 
# 

# %%
def check_gradient(grad, label, last_sorted_grads, plot = False):
    logging.info("CHECKING GRADIENT FOR LABEL {}".format(label))
    sum_abs_grad = np.sum(abs(grad[label]), axis = 0)
    
    current_sorted_grad = (-sum_abs_grad).argsort()
    
#     if len(last_sorted_grads[label]) > 0:
#         for k in [100, 200, 300, 400]:
#             prev_top_k = set(last_sorted_grads[label][-1][:k])
#             current_top_k = set(current_sorted_grad[:k])
#             intersect = prev_top_k.intersection(current_top_k)
#             logging.info('k = {}. How many top Gradients are stable since last epoch?: {}'.format(k, len(intersect)))
        
    for k in [0, 9, 99, 199]:    
        logging.debug('{}th biggest gradient = {}'.format(k, np.sort(-sum_abs_grad)[k]))
    if plot:
        fig = plt.figure(figsize=(30, 1))
        plt.bar(range(sum_abs_grad.shape[0]), sum_abs_grad)
        plt.show()
        print(sum_abs_grad.max(), sum_abs_grad.argmax(), sum_abs_grad.min())

    return current_sorted_grad



#init stuffs
LOAD = False
# LOADPATH = 'TinyCNNreg.conv1.conv2.fc1.fc2.17:09:00'
LOADPATH = 'TinyCNNreg.conv2.fc1.fc2.15:09:24'
# LOADPATH = 'TinyCNNreg.fc1.fc2.16:06:26'
LAST_N_EPOCHS = 10

dataset1 = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = FeedforwardNeuralNetModel(28*28, 128, 10).to(device)
# model = TinyCNN().to(device)
if LOAD:
    model.load_state_dict(torch.load(LOADPATH))
else:

    epochs = 10
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma = 0.7)
    last_sorted_grads = defaultdict(list)
    
    all_rows = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        print("HEREFDAFASDFFDS")
        model.register_gradient()
        model.train()
        target_log  = None # need to record the label to match with the gradient later
        for data, target in train_loader:
            target_log = np.concatenate((target_log, target), axis = 0) if target_log is not None else target
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        CFI_utils.test(model, device, test_loader)
#         scheduler.step()
        grad = CFI_utils.get_grad_each_label(model.gradient_log, 
                                      target_log = target_log, 
                                      layers = ['fc1', 'fc2', 'fc3', 'fc4'], 
                                      labels = range(10))
        torch.save(model.state_dict(), model.model_savename())
        
        row_data = []
        for label in range(10):
            r = []
            logging.info("After {} epoch:".format(epoch))
            last_sorted_grads[label].append(check_gradient(grad, label, last_sorted_grads))
            
            
            if epoch >= LAST_N_EPOCHS:
                for k in [100, 200, 300, 400]:
                    all_top_k = [set(sorted_grad[:k]) for sorted_grad in last_sorted_grads[label][-LAST_N_EPOCHS:]]
                    intersect = set.intersection(*all_top_k)
                    logging.info('k = {}. How many top Gradients are stable among all last {} epochs?: {}'.format(k, LAST_N_EPOCHS, len(intersect)))

# %%



label = 1
for k in [100]:
    all_top_k = [set(sorted_grad[:k]) for sorted_grad in last_sorted_grads[label][-LAST_N_EPOCHS:]]
    intersect_0 = set.intersection(*all_top_k)
    print(intersect_0)
    print(len(intersect_0))

label = 7
for k in [100]:
    all_top_k = [set(sorted_grad[:k]) for sorted_grad in last_sorted_grads[label][-LAST_N_EPOCHS:]]
    intersect_1 = set.intersection(*all_top_k)
    print(intersect_1)
    print(len(intersect_1))
    
print(intersect_0.intersection(intersect_1))

# %% [markdown]
# # Study stable gradients

# %%
layers = ['fc1', 'fc2', 'fc3', 'fc4']
labels = range(10)
K = 25

all_patterns = {}

all_rows = []

for K in [25, 50, 100]:
    print("K=", K)
    row = []
    row.append(str(K))
    for label in labels:
        patterns = []
        #construct the stable gradients 
        all_top_k = [set(sorted_grad[:K]) for sorted_grad in last_sorted_grads[label][-LAST_N_EPOCHS:]]
        intersect = set.intersection(*all_top_k)
        stable_grad = np.array(sorted(list(intersect)))
        print("There are {} stable grad in top K".format(len(stable_grad)))
        print(stable_grad)


        for data, target in train_loader:
            filter_ids = target == label
            data = data[filter_ids]
            logging.debug(data.shape[0])
            pattern = CFI_utils.get_pattern(model, device, data)
            pattern = np.concatenate([pattern[l] for l in layers], axis = 1)
            logging.debug(pattern.shape)
            patterns.append(pattern)

        patterns = np.concatenate(patterns, axis = 0)
        all_patterns[label] = patterns
        logging.info(patterns.shape)
        print("LABEL:", label)
        print("how many unique paths in the full pattern?", np.unique(patterns, axis = 0).shape)
        print("how many unique paths in the filtered pattern?", np.unique(patterns[:, stable_grad ], axis = 0).shape)
        print("how many unique paths in the randomly filtered pattern?", 
              np.unique(patterns[:, 
                                 np.random.choice(458, len(stable_grad), replace = False) ], axis = 0).shape)
        row.append("|".join([str(len(stable_grad)),
                             str(np.unique(patterns, axis = 0).shape[0]),
                             str(np.unique(patterns[:, stable_grad ], axis = 0).shape[0]),
                             str(np.unique(patterns[:, np.random.choice(458, len(stable_grad), replace = False) ], axis = 0).shape[0])
                            ]))
    all_rows.append(",".join(row)+"\n")

with open("gradient_exp_log.csv", "w") as f:
    f.write("K,"+",".join([str(l) for l in labels]) + "\n")
    f.writelines(all_rows)

                   

# %% [markdown]
# # Study stable ReLUs

# %%
all_stable_relus = []
epsilon = 0.001
for label in all_patterns:
    patterns = all_patterns[label]
    print(patterns.shape)
    
    
    relu_sum = np.sum(patterns, axis = 0).squeeze()

    stable_idx = np.concatenate([np.where(relu_sum<=epsilon*patterns.shape[0]), 
                                 np.where(relu_sum>=(1-epsilon)*patterns.shape[0])],
                                axis = 1
                                ).squeeze()
    print("how many unique paths in the filtered pattern?", np.unique(patterns[:, stable_idx ], axis = 0).shape)

    all_stable_relus.append(set(stable_idx))           
    print(stable_idx)
    fig = plt.figure(figsize=(30, 1))
    plt.bar(range(458), relu_sum)
    plt.show()
    print(relu_sum.max())
    
super_stable = set.intersection(*all_stable_relus)
print(super_stable, len(super_stable))

# %%
label = 1

stable_idx = np.array(sorted(all_stable_relus[label]))

patterns = []
for data, target in test_loader:
    filter_ids = target == label
    data = data[filter_ids]
    logging.debug(data.shape[0])
    pattern = CFI_utils.get_pattern(model, device, data)
    pattern = np.concatenate([pattern[l] for l in layers], axis = 1)
    logging.debug(pattern.shape)
    patterns.append(pattern)

test_patterns = np.concatenate(patterns, axis = 0)
logging.info(test_patterns.shape)
print("LABEL:", label)
print("how many unique paths in the full pattern?", np.unique(test_patterns, axis = 0).shape)
print("how many unique paths in the filtered pattern?", np.unique(test_patterns[:, stable_idx ], axis = 0).shape)
print("how many unique paths in the randomly filtered pattern?", 
      np.unique(test_patterns[:, 
                         np.random.choice(458, len(stable_idx), replace = False) ], axis = 0).shape)




train_patterns = all_patterns[label]
raw_train_patterns = np.unique(train_patterns[:, stable_idx], axis = 0)
print(raw_train_patterns.shape)

raw_test_patterns = np.unique(test_patterns[:, stable_idx], axis = 0)
print(raw_test_patterns.shape)

set_train_patterns = set([tuple(p) for p in raw_train_patterns])
set_test_patterns = set([tuple(p) for p in raw_test_patterns])

print(len(set_train_patterns), len(set_test_patterns))
print(len(set_train_patterns.intersection(set_test_patterns)))

# %% [markdown]
# # RQ6:study the hot ReLU and the gradient
# Do at least the following statistics on the activation patterns.
# - What are the top k RELU with the highest gradient value (indicating that they are highly influential ReLU). If we only look at those ReLU, how many activation patterns do we have? This exp should be parametrized by k
# - For activation patterns from images from the same label (e.g 7), what ReLU are stable, and what not? To check for stable, we set a threshold T%. If the value of the ith ReLU is the same for at least T% of the time, we say that ith ReLU is stable.
# - Given that ith ReLU is stable for the label 7, is it also stable for a different label (e.g 9)? Note that for label 7, it may stable and always be 1, but for label 9, it may also be stable but with value 0.
# 

# %% [markdown]
# #Checking ReLUs with highest gradient
# 
# 
print("here")
# %%
def set_relu(layer : torch.nn.Module, neuron : int, always_on = True ): #set the nth relu to be always on or off depending on always_on, true is on
    #params weight, bias
    #first number is #of out put
    with torch.no_grad():
        new_weights = None
        new_bias = None
        for name, param in layer.named_parameters():
            
            if name == "weight":
                new_weights = param.detach()
                new_weights[neuron,:] = torch.zeros(new_weights.shape[1]) #set weights to 0, always
                param.copy_(new_weights)
            elif name == "bias":
                new_bias = param.detach()
                new_bias[neuron] = 1 if always_on else 0 # abs(new_bias[neuron]) if always_on else 0 #EXPLAIN THIS
                param.copy_(new_bias)
        print(new_bias)
        print(new_weights)
        #sd = model.state_dict()
        #sd['weight'] = new_weights
        #sd['bias'] = new_bias
        #layer.load_state_dict(sd)
for x in range(16):
    set_relu(model.fc3,np.random.randint(0,64), always_on=False)      

print("updated")
[print("{}: {}".format(name,param))  for name,param in model.fc2.named_parameters()  ]

torch.onnx.export(model, torch.ones(784),  "l3disabled16.onnx")
print("complete")