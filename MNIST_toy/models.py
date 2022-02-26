import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import logging
from datetime import datetime
import copy
# to get activation
ACTIVATION = None

def get_activation(name, tensor_logger, detach):
    if detach:
        def hook(model, input, output):
            raw = torch.sigmoid(torch.flatten(
                output, start_dim=1, end_dim=-1)).cpu().detach().numpy()
            raw = raw > 0.5 #need to convert here, because float takes a lot more memory
            logging.debug("{}, {}".format(name,raw.shape))
            tensor_logger[name] = np.concatenate((tensor_logger[name], raw), 
                                                axis = 0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)
            # ACTIVATION = np.concatenate((ACTIVATION, raw), axis=1)
        return hook
    else:
        #keep the gradient, so cannot convert to bit here
        def hook(model, input, output):
            raw = torch.sigmoid(torch.flatten(
                output, start_dim=1, end_dim=-1))
            logging.debug("{}, {}".format(name,raw.shape))
            tensor_logger[name] = torch.cat((tensor_logger[name], raw), 
                                                axis = 0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)
            # ACTIVATION = np.concatenate((ACTIVATION, raw), axis=1)
        return hook

def get_gradient(name, gradient_logger, detach):
    def hook(model, grad_input, grad_output):
        raw = grad_output
        assert(len(raw)==1)
        raw = raw[0].cpu().detach().numpy()
        gradient_logger[name] = np.concatenate((gradient_logger[name], raw), axis = 0) if name in gradient_logger else raw
        
    return hook


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.tensor_log = {}
        self.gradient_log = {}
        self.hooks = []
        self.bw_hooks = []

    def reset_hooks(self):
        self.tensor_log = {}
        for h in self.hooks:
            h.remove()
            
    def reset_bw_hooks(self):
        self.input_labels = None
        self.gradient_log = {}
        for h in self.bw_hooks:
            h.remove()
            
    def register_log(self, detach):
        raise NotImplementedError

    def register_gradient(self, detach):
        raise NotImplementedError
        
    def model_savename(self):
        raise NotImplementedError
        
    def get_pattern(self, input, layers, device, flatten = True):
        self.eval()
        self.register_log()
        self.forward(input.to(device))
        tensor_log = copy.deepcopy(self.tensor_log)
        if flatten:
            return np.concatenate([tensor_log[l] for l in layers], axis=1)
        return tensor_log
    

class TinyCNN(BaseNet):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1152, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
#         x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.conv1.register_forward_hook(get_activation('conv1', self.tensor_log, detach)))
        self.hooks.append(self.conv2.register_forward_hook(get_activation('conv2', self.tensor_log, detach)))
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))

    def model_savename(self, tag=""):
        return "TinyCNN"+tag+datetime.now().strftime("%H:%M:%S")

class FeedforwardNeuralNetModel(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    def register_log(self, detach=True):
        self.reset_hooks()    
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))
        self.hooks.append(self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach)))
        self.hooks.append(self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach)))
        
    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc4.register_backward_hook(get_gradient('fc4', self.gradient_log, detach)))
        
    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28*28)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = F.log_softmax(out, dim=1)
        return out
    def model_savename(self):
        return "FFN"+datetime.now().strftime("%H-%M-%S")

class PatternClassifier(BaseNet):
    def __init__(self, input_dim, max_unit, output_dim):
        super(PatternClassifier, self).__init__()
        self.max_unit = max_unit
        # Linear function
        self.fc1 = nn.Linear(self.max_unit, output_dim, bias = False)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
    def register_log(self):
        self.reset_hooks()    
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log)))
        # self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log)))
    def forward(self, x):
        # out = F.relu(self.fc1(x))
        out = self.fc1(x[:, :self.max_unit])
        # out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
    def model_savename(self):
        return "PatternClasffier"+datetime.now().strftime("%H:%M:%S")

def fgsm_attack(image, eps, data_grad):
    sign_data_grad = data_grad.sign()
    pertubed_image = image + eps*sign_data_grad
    pertubed_image = torch.clamp(pertubed_image, 0, 1)
    return pertubed_image
