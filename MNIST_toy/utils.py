from __future__ import print_function
import argparse
import seaborn as sns
from tqdm import tqdm
import copy
import numpy as np
from torchvision import datasets
from sklearn.decomposition import PCA
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from typing import Tuple
import logging

class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, patterns, targets,):
        self.inputs = inputs
        self.patterns = patterns
        self.targets = targets
        assert(self.patterns.shape[0] == self.targets.shape[0])
    def __len__(self):
        return self.patterns.shape[0]

    def __getitem__(self, idx):
        return self.patterns[idx], self.targets[idx]

def bit_diff(a, b):
    return np.logical_xor(a, b).sum()    
    
def get_pattern_dataset(original_dataset, new_targets, model, layers, train_kwargs):
    loader = torch.utils.data.DataLoader(original_dataset, **train_kwargs)
    patterns = test(model, torch.device('cuda'), loader, trace=True)
    pattern_dataset = PatternDataset(np.concatenate([patterns[l] for l in layers], axis = 1), targets = new_targets)
    return pattern_dataset

def get_single_label_dataset(label):
    dataset = datasets.MNIST(root='/home/nle/workspace/CFI_NNs/data')
    idx = dataset.targets==label
    dataset.new_targets = dataset.targets[idx]
    dataset.new_data = dataset.data[idx]
    return dataset

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
def sampling_inside_ball(n_samples, n_dim, r, p=2, origin=None, algo=2):
    """
    sampling N points of k-dimension inside a ball of l-p norm of radius r
    algo1: simply sample a point, and check if it is inside the ball
    algo2: based on https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html
    - step 1: generate n_dim vectors, then divided by their norms. We get a bunch of vectors on the surface of the ball
    - step 2: scales those vectors uniformly so they are of uniformly distributed distances to the center
    """
    res = []
    if origin is None:
        origin=np.zeros(n_dim)
    if algo==1:
        print(origin.shape)
        tries = 0
        while len(res)<n_samples and tries < n_samples*100:
            tries+=1
            new_vec = np.random.rand(n_dim)
            norm = np.linalg.norm(new_vec, ord=p)

            if norm<r:
                res.append(new_vec+origin)
            else:
                print(norm)
    elif algo==2:
        for _ in range(n_samples):
            new_vec = np.random.normal(0, 1, n_dim)
            norm = np.linalg.norm(new_vec, ord=p)
            new_vec = new_vec/norm
            
            d = np.random.random() ** (1/n_dim)
            new_vec = new_vec * d * r
            logging.debug(new_vec[:10])
            logging.debug(origin[:10])
            logging.debug(np.add(new_vec, origin)[:10])
            res.append(np.add(new_vec, origin))
            
    return res
        
        
def get_pattern(model, device, input)->dict:
    """
    Run one input through model and record activation pattern
    """
    model.eval()
    model.register_log()
    model(input.to(device))
    tensor_log = copy.deepcopy(model.tensor_log)

    return tensor_log
    
def test(model, device, test_loader, trace=False, detach = True):
    model.eval()
    test_loss = 0
    correct = 0
    if trace:
        model.register_log(detach)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if not trace:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    tensor_log = copy.deepcopy(model.tensor_log)
        
    model.reset_hooks()
    return tensor_log

def fit_pca(all_activations, layers, n = 2):
    filtered = [all_activations[l] for l in layers]
    stacked = np.concatenate(filtered, axis=1)
    print("fit_pca:", stacked.shape)
    pca = PCA(n_components=n)
    return pca.fit(stacked)


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

def plot_pca(model, data_loader, pca, layers, device = torch.device('cuda')):
    plt.figure()
    plt.xlim(-4, 4)
    plt.ylim(-4,4)
    colors = sns.color_palette("tab10")
    Cs = []
    model.register_log()
    for data, target in data_loader:
        Cs.extend([colors[t] for t in target])
        data = data.to(device).float()
        model(data)

    tensor_log = copy.deepcopy(model.tensor_log)
    Xs_train = pca.transform(np.concatenate([tensor_log[l] for l in layers], axis=1))
    plt.scatter(Xs_train[:,0], Xs_train[:,1],c = Cs, s = 2)  

def plot_pca_3d(model, data_loader, pca, layers, device = torch.device('cuda')):
    plt.figure()
    plt.xlim(-4, 4)
    plt.ylim(-4,4)
    colors = sns.color_palette("tab10")
    Cs = []
    model.register_log()
    for data, target in data_loader:
        Cs.extend([colors[t] for t in target])
        data = data.to(device)
        model(data)

    tensor_log = copy.deepcopy(model.tensor_log)
    Xs_train = pca.transform(np.concatenate([tensor_log[l] for l in layers], axis=1))
    plt.scatter(Xs_train[:,0], Xs_train[:,1], Xs_train[:, 2], c = Cs, s = 2)  

def fgsm_attack(image, eps, data_grad):
    sign_data_grad = data_grad.sign()
    pertubed_image = image + eps*sign_data_grad
    pertubed_image = torch.clamp(pertubed_image, 0, 1)
    return pertubed_image

def sample_pair(data_loader)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a random image from data_loader, then find the image closest to it in the same data_loader
    """
    inputs, labels = next(iter(data_loader))
    sample = inputs[0]
    label = labels[0]
    print(data_loader.batch_size)
    print(sample.shape)
    sample_tiled = sample.unsqueeze(0).repeat(data_loader.batch_size, 1, 1, 1)
    print(sample_tiled.shape)
    
    min_dist = 999999
    closest_sample = None
    for inputs, labels in data_loader:
        dist = torch.norm(inputs - sample_tiled, dim=(2,3)).squeeze()
        local_min_dist, idx = torch.min(dist, dim = 0)
        
        if local_min_dist < min_dist and local_min_dist > 0:
            closest_sample = inputs[idx]
            min_dist = local_min_dist    
    return sample, closest_sample, min_dist



def gen_adv( model, device, test_loader, epsilon ) -> Tuple[float, float]:

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, data) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, data.squeeze().detach().cpu().numpy() ))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

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

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")



    train_patterns = test(model, device, train_loader, trace=True)
    test_patterns = test(model, device, test_loader, trace=True)

    train_bitmap = binarize_trace(train_patterns)
    test_bitmap = binarize_trace(test_patterns)

    train_bitmap_set = set(train_bitmap)
    test_bitmap_set = set(test_bitmap)

    # print(train_bitmap[:10])
    # print(test_bitmap[:10])

    print("len train bitmap", len(train_bitmap))
    print("n unique in train bitmap", len(train_bitmap_set))

    print("len test bitmap", len(test_bitmap))
    print("n unique in test bitmap", len(test_bitmap_set))

    print("test bitmap is subset of train bitmap?",
          test_bitmap_set.issubset(train_bitmap_set))
    print("interesection size:", len(
        test_bitmap_set.intersection(train_bitmap_set)))

    pca = pca(train_bitmap)

    if args.dump_bitmap:
        with open("train_bitmap", "w") as f:
            for encode in tqdm(train_bitmap):
                f.write(" ".join([str(n) for n in encode]) + "\n")
        with open("test_bitmap", "w") as f:
            for encode in tqdm(test_bitmap):
                f.write(" ".join([str(n) for n in encode]) + "\n")


if __name__ == '__main__':
    main()
