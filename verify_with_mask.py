import arguments
from batch_branch_and_bound import relu_bab_parallel
from bab_verification_general import config_args
from beta_CROWN_solver import LiRPAConvNet

import auto_LiRPA
import models
import torch
from torchvision import datasets, transforms
from torchsummary import summary

LOADPATH = 'MNIST_toy/FFN18_28_21'
DUMMY_TARGET = torch.empty((1, 28, 28))

use_cuda = True
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

LOADED = False

class LinearModel(torch.nn.Module):
    def __init__(self, n_output):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(4, 3)
        self.fc1 = torch.nn.Linear(3, 5)
        self.fc2 = torch.nn.Linear(5, n_output)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.relu(self.fc0(out))
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
 
def verify(data_loader):
    eps = arguments.Config["specification"]["epsilon"]
    device = torch.device('cuda')

    if LOADED: #verifying MNIST. Not interesting for now.
        N_OUTPUT = 10
        model = models.FeedforwardNeuralNetModel(28*28, 128, 10)
        model.load_state_dict(torch.load(LOADPATH,
                              map_location=device))
        model.to(device)
        summary(model, (1, 28, 28))

        x, _ = next(iter(data_loader))
        x = x[0]

        y = _[0].item()
        print(y)

        test = (y+1) % N_OUTPUT
        print("Trying to verify that the network will never predict {}".format(test))

        # assert 10 > 1
        # we only support c with shape of (1, 1, n)
        c = torch.zeros((1, 1, N_OUTPUT), device=device)
        c[0, 0, y] = 1
        c[0, 0, test] = -1

        wrapped_model = LiRPAConvNet(
            model, pred=y, test=None, in_size=(1, 28, 28), device='cuda', c=c)
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
            [data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1).unsqueeze(0).cuda()
        print("domain", domain.shape)

        res = relu_bab_parallel(wrapped_model, x=x, domain=domain)

        print(res)

    else: #verifying a super small network. Much better to understand things
        N_OUTPUT = 4
        model = LinearModel(N_OUTPUT)
        model.to(device)
        summary(model, (1, 4))

        x = (torch.rand((1, 4)) - 0.5).to(device)
        print("X", x)
        print(model(x))
        y = torch.argmax(model(x)[0]).cpu().item()

        test = (y+1)%N_OUTPUT

        print("Trying to verify that the network will never predict {} upon seeing {}".format(test, y))
        c = torch.zeros((1, 1, N_OUTPUT), device=device)
        c[0, 0, y] = 1
        c[0, 0, test] = -1
        print("C", c)
        wrapped_model = LiRPAConvNet(
            model, pred=y, test=None, in_size=(1, 4), device='cuda', c=c)
        data_ub = x + eps
        data_lb = x - eps

        ball = auto_LiRPA.PerturbationLpNorm(
            eps=eps, x_L=data_lb.to(device), x_U=data_ub.to(device))

        x = x.unsqueeze(0).to(device)
        x = auto_LiRPA.BoundedTensor(x, ball)
        domain = torch.stack(
            [data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1).unsqueeze(0).cuda()
        print("domain", domain.shape)

        res = relu_bab_parallel(wrapped_model, x=x, domain=domain)

        print(res)

if __name__ == "__main__":
    config_args()
    verify(test_loader)
