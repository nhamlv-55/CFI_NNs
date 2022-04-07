import arguments
from batch_branch_and_bound import relu_bab_parallel
from bab_verification_general import config_args
from beta_CROWN_solver import LiRPAConvNet

import auto_LiRPA
import models
import torch
from torchvision import datasets, transforms
from torchsummary import summary
#from matplotlib import pyplot as plt
import numpy as np
import json

LOADPATH = 'MNIST_toy/FFN18_28_21'
DUMMY_TARGET = torch.empty((1, 28, 28))
NP_FILE_PATH="np_for_"
IMAGE_FILE_PATH="an_image_for_"
use_cuda = False
batch_size = 32
test_batch_size = 32
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

dataset2 = datasets.MNIST('data', train=False,
                          transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

LOADED = True


class LinearModel(torch.nn.Module):
    def __init__(self, n_output):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(4, 8)
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, n_output)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.relu(self.fc0(out))
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def get_some_images(data_loader):
    for label in range(10):
        while True:
            x, _ = next(iter(data_loader))
            x = x[0]

            y = _[0].item()
            print(y)
            if y == label:
                xnp = x.numpy()
                np.save("{}{}.npy".format(NP_FILE_PATH,y), xnp)
                plt.imshow(xnp.reshape(28,28))
                plt.savefig("{}{}.png".format(IMAGE_FILE_PATH, y))                     
                break

    return

def get_fixed_relu_mask(label, eps, file):
    network = {'fc1': 256, 'fc2': 128, 'fc3': 64, 'fc4': 10}

    results = [[[],[]],
                [[],[]],
                [[],[]],
                [[],[]]]
    with open(file, "r") as f:
        raw_mask = json.load(f)
    mask = raw_mask[eps][label]
    for counter, stable_idx in enumerate(mask["stable_idx"]):
        if stable_idx < 256:
            results[0][0].append(stable_idx)

            results[0][1].append(1.0 if mask["alpha_pattern"][counter] else -1.0)
        elif stable_idx>=256 and stable_idx<256+128:
            results[1][0].append(stable_idx - 256)
            results[1][1].append(1.0 if mask["alpha_pattern"][counter] else -1.0)
        elif stable_idx>=256+128 and stable_idx < 256 + 128 + 64:
            results[2][0].append(stable_idx - 256 - 128)
            results[2][1].append(1.0 if mask["alpha_pattern"][counter] else -1.0)


    branching_decision = []
    coeffs = []
    for l_idx, layer in enumerate(results):
        for r_idx, relu in enumerate(layer[0]):
            branching_decision.append([l_idx, relu])
            coeffs.append([layer[1][r_idx]])


    print(results)
    return {"decision": branching_decision, "coeffs": coeffs}

def verify(y, test, should_fix_relu):
    eps = arguments.Config["specification"]["epsilon"]
    device = torch.device('cpu')

    if LOADED:  # verifying MNIST. Not interesting for now.
        N_OUTPUT = 10
        model = models.FeedforwardNeuralNetModel(28*28, 128, 10)
        model.load_state_dict(torch.load(LOADPATH,
                              map_location=device))
        model.to(device)
        summary(model, (1, 28, 28))

        x = torch.from_numpy(np.load("{}{}.npy".format(NP_FILE_PATH, y))).to(device)

        print("Trying to verify that the network will never predict {} upon seeing {}".format(test, y))

        # assert 10 > 1
        # we only support c with shape of (1, 1, n)
        c = torch.zeros((1, 1, N_OUTPUT), device=device)
        c[0, 0, y] = 1
        c[0, 0, test] = -1

        """setup the fixed relu split"""
        if should_fix_relu:
            fixed_relu_mask = get_fixed_relu_mask(str(y), "0.0005", "MNIST_toy/relu_exp_data16-06-03.json")
        else:
            fixed_relu_mask = None
        # return
        wrapped_model = LiRPAConvNet(
            model, pred=y, test=None, in_size=(1, 28, 28), device='cpu', c=c, fixed_relu_mask=fixed_relu_mask)
        data_ub = x + eps
        data_lb = x - eps

        ball = auto_LiRPA.PerturbationLpNorm(
            eps=eps, x_L=data_lb.to(device), x_U=data_ub.to(device))
        if list(wrapped_model.net.parameters())[0].is_cuda:
            print("cuda")
            x = x.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        print('Model prediction is:', wrapped_model.net(x))
        x = x.unsqueeze(0).to(device)
        print(x.shape)
        x = auto_LiRPA.BoundedTensor(x, ball)
        domain = torch.stack(
            [data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1).unsqueeze(0)
        print("domain", domain.shape)

        res = relu_bab_parallel(wrapped_model, x=x, domain=domain)

        print(res)
        lower_bound = res[0]
        solving_time = res[-1]
        with open("results.csv", 'a') as f:
            f.writelines(["{},{},{},{},{},{}\n".format(should_fix_relu, y, test, eps, solving_time, lower_bound)])


    else:  # verifying a super small network. Much better to understand things
        N_OUTPUT = 10
        model = LinearModel(N_OUTPUT)
        model.to(device)

        """ Nham: uncomment to generate a new model and input"""
        # torch.save(model.state_dict(), "model_for_study_ABC.pt")
        # x = (torch.rand((1, 4)) - 0.5).to(device)
        model.load_state_dict(torch.load("model_for_study_ABC.pt"))

        summary(model, (1, 4))

        x = torch.tensor([[0.3144, -0.4164, -0.1399, -0.3196]]).to(device)
        print("X", x)
        y = torch.argmax(model(x)[0]).cpu().item()

        test = (y+1) % N_OUTPUT

        print("Trying to verify that the network will never predict {} upon seeing {}".format(test, y))
        c = torch.zeros((1, 1, N_OUTPUT), device=device)
        c[0, 0, y] = 1
        c[0, 0, test] = -1
        print("C", c)

        """setup the fixed relu split"""
        fixed_relu_mask = [[[0, 3, 5, 2], [1, -1, 1, -1]],
                            [[1,4], [1, -1]]]

        wrapped_model = LiRPAConvNet(
            model, pred=y, test=None, in_size=(1, 4), device='cpu', c=c, fixed_relu_mask=fixed_relu_mask)
        data_ub = x + eps
        data_lb = x - eps

        ball = auto_LiRPA.PerturbationLpNorm(
            eps=eps, x_L=data_lb.to(device), x_U=data_ub.to(device))

        x = x.unsqueeze(0).to(device)
        x = auto_LiRPA.BoundedTensor(x, ball)
        domain = torch.stack(
            [data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1).unsqueeze(0)
        print("domain", domain.shape)

        res = relu_bab_parallel(wrapped_model, x=x, domain=domain)

        print(res)

        lower_bound = res[0]
        solving_time = res[-1]
        with open("results.csv", 'a') as f:
            f.writelines(["{},{},{},{},{},{}\n".format(should_fix_relu, y, test, eps, solving_time, lower_bound)])

if __name__ == "__main__":
    config_args()
    for i in range(1, 10):
        for j in range(10):
            if i!=j:
                verify(i, j, should_fix_relu=True)
                verify(i, j, should_fix_relu=False)
