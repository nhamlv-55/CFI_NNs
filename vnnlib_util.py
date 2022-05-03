from cgi import test
from complete_verifier.read_vnnlib import * #import parser
import torch
import torchvision
import torchvision.datasets 
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms
import pandas as pd

def find_lp_from_image(image : torch.Tensor, images : list, epsilon  : float,  p = "inf" ) -> list: #find all images within epsilon of base image
    distances = []
    
    metric = None
    if p == "inf":
        metric = torch.max
    else:
        def norm(x : torch.Tensor):
            return torch.norm(x, p = p)
        metric = norm

    for i in images:
        distances += [float(metric(  (image-i).flatten() ) )]
        #distances += [float(torch.norm(  (image-i).flatten(),p=p ) )]

    distances.sort()
    return distances



def label_data(labels : list , data : list) -> dict: #put data in a dictionary with labels
    result = dict(zip(list(range(10)), [ [] for x in range(10)] ))
    for i in range(len(labels)):
        result[labels[i]] += [data[i]]
    return result


x = read_vnnlib_simple('vnncomp2021/benchmarks/mnistfc/prop_9_0.05.vnnlib',784,10)

trns_norm = torchvision.transforms.ToTensor()

#SEED FOR COMPETITION
#67836002566492401312858690134950222230020534910151188007475391202415929811459 

prop_to_class = dict(  zip(list(range(15))   , [8, 1,0,7,2,9,3,9,6,8,0,0,9,9,1 ]   ))

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trns_norm)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=trns_norm)
#mnist_trainset.data.shape
#mnist_trainset.train_labels

training_data_with_labels = dict()
torch.random.manual_seed(int(67836002566492401312858690134950222230020534910151188007475391202415929811459 % 1000000000 ))

loader_test = DataLoader(mnist_testset, batch_size=10000,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))

test_data, test_labels = next(iter(loader_test))
#test_labels = [int(x) for x in mnist_testset.test_labels]
#test_data = list(mnist_testset.data)
test_data = list(test_data)
test_labels = list(map( int , test_labels))

test_data = test_data[0:13] + test_data[14:] #for some reason remove the 14th entry ?????
test_labels = test_labels[0:13] + test_labels[14:]


test_data = [x.resize(28,28) for x in test_data]

test_labelled_data = label_data(test_labels, test_data)

class_distances = dict()

p = "inf"

for i in range(max(set(test_labels[:15]))+1):
    if i in test_labels[:15]:
        class_distances[i] = find_lp_from_image(test_labelled_data[i][0], test_labelled_data[i], epsilon = 0.05, p = p)[:900]

out = pd.DataFrame(class_distances)
out.to_csv("class_distances_l_{}.csv".format(p))

print("what is my purpose? You're a breakpoint")
