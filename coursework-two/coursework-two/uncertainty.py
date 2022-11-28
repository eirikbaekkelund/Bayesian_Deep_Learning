import numpy as np; import torch
import matplotlib.pyplot as plt

import torch.distributions as dist
import torch.nn as nn

dataset, validation_set = torch.load("two_moons.pt")
X_train, y_train = dataset.tensors

class TwoMoonsNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 100),
                        nn.ReLU(), 
                        nn.Linear(100, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
        
    def forward(self, x):
        h = self.net(x)
        return torch.sigmoid(h).squeeze(1)
    
network = TwoMoonsNetwork()

def log_likelihood(network, X, y):
    """
    This function computes the log probability `log p(y | x, theta)`
    for a batch of inputs X.
    
    INPUT:
    network : instance of classifier network, extends `nn.Module`
    X       : batch of inputs; torch.FloatTensor, matrix of shape = (batch_size, 2)
    y       : batch of targets: torch.FloatTensor, vector of shape = (batch_size,)
    
    OUTPUT:
    lp      : log probability value of log p(y | x, theta); scalar
    
    """
    print(network.forward(X))
    return dist.Bernoulli(network.forward(X)).log_prob(y).sum(0)

def log_prior(network):
   
    params = nn.utils.parameters_to_vector([w for w in network.parameters()])
    return dist.MultivariateNormal(loc=torch.zeros(params.shape[0], scale=torch.eye(params.shape[0]))).log_prob(params).sum(0)

def log_joint_minibatch(network, X_batch, y_batch, N_training):
    """ Return a minibatch estimate of the full log joint probability 
    
    INPUT:
    network    : instance of classifier network, extends `nn.Module`
    X_batch    : batch of inputs; torch.FloatTensor, matrix of shape = (batch_size, 2)
    y_batch    : batch of targets: torch.FloatTensor, vector of shape = (batch_size,)
    N_training : total number of training data instances in the full training set

    OUTPUT:
    lp : return an estimate of log p(y, theta | X), as computed on the batch; scalar.

    """
    M = X_batch.shape[0]
    
    return (N_training / M) * (log_prior(network) + log_likelihood(network, X_batch, y_batch))

with torch.no_grad():
    full_data_lp = log_prior(network) + log_likelihood(network, X_train, y_train)
print( "Full data log probability: %0.4f" % full_data_lp.item() )
