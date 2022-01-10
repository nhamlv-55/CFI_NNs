from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
import numpy as np
from torchviz import make_dot


from sklearn.decomposition import PCA


path_penalty = .01
p = 2

class LoggerLayer(nn.Module):
    def __init__(self, other_layer: nn.Module, log: list):
        super(LoggerLayer, self).__init__()
        self.log = log
        self.layer = other_layer 
        self.logging = True  # now we log during traiing

    def forward(self, x):
        y = self.layer(x)
        if self.logging:
            self.log.append(y)
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1152, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x, trace_mode=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # 784 down to 256 -> activation -> 256 down to 32 -> activation -> 32 down to 10 -> activation
        # Linear function
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function (readout)
        self.fc3 = nn.Linear(32, output_dim)

        # loggers
        self.log_relu = nn.ReLU()
        self.logger1 = LoggerLayer(self.log_relu, [])
        self.logger2 = LoggerLayer(self.log_relu, [])
        self.logger3 = LoggerLayer(self.fc3, [])
        self.d1 = nn.Dropout(p=0.25)
        self.d2 = nn.Dropout(p=0.25)
    def forward(self, x):
        #print(x.shape)
        
        #out = self.d2(x)
       
        out = self.fc1(x)
        
        out = self.logger1(out)
        #out = self.d1(out)
        out = self.fc2(out)
        out = self.logger2(out)

        out = self.logger3(out)

        #out = F.log_softmax(out, dim=1)
        return out
    



# to get activation
ACTIVATION = None


def get_activation(name):
    def hook(model, input, output):
        global ACTIVATION
        raw = torch.sigmoid(torch.flatten(
            output, start_dim=1, end_dim=-1)).cpu().detach().numpy()
        # raw = raw > 0 #need to convert here, because float takes a lot more memory
        #print(name, raw.shape)
        ACTIVATION = np.concatenate((ACTIVATION, raw), axis=1)
    return hook

def custom_loss(output, target):
   
    
    return torch.nn.functional.cross_entropy(output, target) 

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    global path_penalty
    global p

    
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data = data.view(-1, 28 * 28)
        data = data.to(torch.float)
       
        #print("data is: {}".format(data.size()))
        data, target = data.to(device), target.to(device)
        #print(target[0])
        optimizer.zero_grad()
       
        model.logger1.log = []
        model.logger2.log = []
        model.logger3.log = []
       
        output = model(data)
        make_dot(output, params = dict(model.named_parameters())).render("torchviz", format="png")
        #loss = F.nll_loss(output, target)
        #loss = custom_loss(output, target)

        l1_lambda = 0.0005
        l1_penalty = torch.sum(torch.abs(model.fc1.weight)) +  torch.sum(torch.abs(model.fc2.weight)) +torch.sum(torch.abs(model.fc3.weight))

        full_paths = []
        for i in range(len(model.logger1.log)):
            #print(model.logger1.log[i].size())
            #print(model.logger2.log[i].size())
            #print("whole log is {}".format(model.logger1.log.size()))
            full_paths.append(torch.cat((model.logger1.log[i]  , model.logger2.log[i] , model.logger3.log[i]  ),1))
            
        
        path_loss = 0
        #print("{} paths".format(len(full_paths)))
       
        full_paths = torch.split(full_paths[0], full_paths[0].size()[0], dim = 0)[0]
        #print("{} paths".format(len(full_paths)))
       
        if path_penalty != 0:
            
            for i,path1 in enumerate(full_paths):
                for j,path2 in enumerate(full_paths):
                    #print(path1.size())
                    if target[i] == target[j] and i != j and i < j:
                        #binarized1 = torch.abs(torch.minimum(torch.ceil(path1),torch.ones(path1.size()))).unsqueeze(0) #binarize activations
                        #binarized2 = torch.abs(torch.minimum(torch.ceil(path2),torch.ones(path2.size()))).unsqueeze(0)
                        #print("first binarized is {}".format(binarized1))
                        #print("second binarized is {}".format(binarized2))
                        #diff = torch.abs(binarized1-binarized2)#hamming distance
                        #print("differences are {}".format(diff))
                        #diff = torch.sum(torch.cdist(binarized1, binarized2, p = 1))
                        #path_loss += diff
                        #path_loss += torch.cdist(path1.unsqueeze(0),path2.unsqueeze(0),p=p)
                        path_loss += torch.nn.functional.cosine_similarity(path1, path2, dim = 0)
                        #print(diff)


        #loss +=  l1_lambda * l1_penalty
        #print(path_loss)
        #print("\nold loss{}".format(loss))
        #loss += path_penalty * path_loss
        loss = path_penalty * path_loss +  custom_loss(output, target)
        #print("new loss{}".format(loss))

        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, trace=False):
    global ACTIVATION
    all_activation = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #print('--------------')

            data = data.view(-1, 28 * 28)
            data, target = data.to(device), target.to(device)
            ACTIVATION = np.zeros((data.shape[0], 1), dtype=float)
            output = model(data)
            
            if trace:
                all_activation.append(copy.deepcopy(ACTIVATION))
            # sum up batch loss
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("before test accuracy")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_activation


def make_pca(all_activations):
    #stacked = np.concatenate(all_activations, axis=0)
    pca = PCA(n_components=2)
    pca = pca.fit(all_activations)
    return pca


def binarize_trace(all_activations):
    """
    convert all activations into a list of bitmap
    all_activations is a list of ACTIVATION
    an ACTIVATION is a of the form {'conv1': 1000*conv1_size, 'conv2': 1000*conv2_size, etc.} 
    (1000 is the batch size, set in args.batch_size)
    """
    stacked = np.concatenate(all_activations, axis=0)

    stacked = stacked > 0.5
    results = []
    for bitmap in stacked:
        results.append(tuple(bitmap))
        # results.append(tuple(np.packbits(bitmap)))
    print(results[0])
    return results


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--dump-bitmap', action='store_true', default=False,
                        help='store the bitmap')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--p', type=float,default=2,help='Value of p in p-norm for path based regularization, can be float')
    parser.add_argument('--lambda-pr', type=float, default = .01, help ='lambda for path based regularization')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("cuda: {}".format(use_cuda))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    path_penalty = args.lambda_pr
    p = args.p

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
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = FeedforwardNeuralNetModel(28*28, 128, 10).to(device)
    print(model)
    print(model.fc1.weight.size())
    
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print("batch_size: {}, lr: {}, path lambda: {}, p: {}".format(args.batch_size, args.lr,path_penalty,p))
    print("epochs are {}".format(args.epochs)) 
  
  
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # register trace
    # first layer should not make any difference?
    # model.conv1.register_forward_hook(get_activation('conv1'))
    # model.conv2.register_forward_hook(get_activation('conv2'))
    model.fc1.register_forward_hook(get_activation('fc1'))
    model.fc2.register_forward_hook(get_activation('fc2'))

    train_patterns = test(model, device, train_loader, trace=True)
    test_patterns = test(model, device, test_loader, trace=True)

    train_bitmap = binarize_trace(train_patterns)
    test_bitmap = binarize_trace(test_patterns)

    train_bitmap_set = set(train_bitmap)
    test_bitmap_set = set(test_bitmap)

    # print(train_bitmap[:10])
    # print(test_bitmap[:10])\
    print("batch_size: {}, lr: {}, path lambda: {}, p: {}".format(args.batch_size, args.lr,path_penalty,p))
    print("epochs are {}".format(args.epochs)) 

    print("len train bitmap", len(train_bitmap))
    print("n unique in train bitmap", len(train_bitmap_set))

    print("len test bitmap", len(test_bitmap))
    print("n unique in test bitmap", len(test_bitmap_set))

    print("test bitmap is subset of train bitmap?",
          test_bitmap_set.issubset(train_bitmap_set))
    print("interesection size:", len(
        test_bitmap_set.intersection(train_bitmap_set)))

    pca = make_pca(train_bitmap)

    if args.dump_bitmap:
        with open("train_bitmap", "w") as f:
            for encode in tqdm(train_bitmap):
                f.write(" ".join([str(n) for n in encode]) + "\n")
        with open("test_bitmap", "w") as f:
            for encode in tqdm(test_bitmap):
                f.write(" ".join([str(n) for n in encode]) + "\n")


if __name__ == '__main__':
    main()
