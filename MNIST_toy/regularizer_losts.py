import torch

import torch.nn as nn

import logging


def l1(model, layers = None):
    """
    layers: a list of layers to calculate l1, e.g [model.conv1, model.conv2, model.fc2]
    
    """
    l1_norm = 0
    if layers is None:
        #take l1 of all param
        l1_norm = sum(p.detach().abs().sum() for p in model.parameters())
    else:
        for l in layers:
            l1_norm += sum(p.detach().data.abs().sum() for p in l.parameters())
    return l1_norm

def batch_activation_diff(tensor_log, target, layers):
    activation_diff = 0
    tensor_log_flatten = torch.cat([tensor_log[l] for l in layers], axis = 1)
    logging.debug(tensor_log_flatten.shape)
    all_paths = torch.split(tensor_log_flatten, 1)
    logging.debug(len(all_paths))
    logging.debug(all_paths[0])
    for i, path_i in enumerate(all_paths):
        for j, path_j in enumerate(all_paths[i+1:]):
            if target[i] == target[j]:
                logging.debug(path_i)
                logging.debug(path_j)
                diff =torch.nn.functional.cosine_similarity(path_i, path_j, dim=1)
                logging.debug("diff: {}, {}".format(diff, diff.shape))
                activation_diff+=diff
    return -activation_diff
#test
if __name__=="__main__":
    #TEST L1
    
    from models import FeedforwardNeuralNetModel, TinyCNN, PatternClassifier
    device = torch.device("cuda")

    LOADPATH = 'TinyCNN15:36:27'

    model = TinyCNN().to(device)
    
    model.load_state_dict(torch.load(LOADPATH))
    
    assert(l1(model)==l1(model, [model.conv1, model.conv2, model.fc1, model.fc2]))
    
    