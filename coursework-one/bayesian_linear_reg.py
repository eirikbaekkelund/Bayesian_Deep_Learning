import sys
sys.settrace
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
os.path.a

def predict_probs_MAP(Phi, w):
    """
    Given a "design matrix" Phi, and a point estimate w, compute p(y = 1 | Phi, w)
    
    INPUT:
    Phi   : (N, D) tensor of input features, where N is the number of 
            observations and D is the number of features
    w     : (D,) vector of weights

    OUTPUT:
    y_hat : (N,) vector of probabilities p(y=1 | Phi, w)
    """
    return torch.sigmoid(Phi @ w)

def log_joint(Phi, y, w, sigma=10):
    """
    Compute the joint probability of the data and the latent variables.
    
    INPUT:
    Phi   : (N, D) tensor of input features, where N is the number of 
            observations and D is the number of features
    y     : (N,) vector of outputs (targets). Should be a `torch.FloatTensor`
            containing zeros and ones
    w     : (D,) vector of weights
    sigma : scalar, standard deviation of Gaussian prior distribution p(w).
            Leave this set to sigma=10 for purposes of this exercise

    OUTPUT:
    log_joint : the log probability log p(y, w | Phi, sigma), a torch scalar
        
    """
    log_prior_w = dist.MultivariateNormal(loc=torch.zeros(Phi.shape[-1]), covariance_matrix=sigma*torch.eye(Phi.shape[-1]) ).log_prob(w)
    
    y_hat = predict_probs_MAP(Phi,w)
    log_prob_y = torch.FloatTensor(sum([dist.Bernoulli(torch.FloatTensor.float((y_hat[i]))).log_prob(torch.FloatTensor.float( y[i]),  ) for i in range(len(y_hat))]) )
    print(log_prob_y)
    print(log_prior_w)
    return log_prob_y + log_prior_w

# def loss(weights, Phi, y):
#     return torch.sum(y @ torch.log(torch.where(torch.sigmoid(Phi @ weights) > 0.5, 1, 0))  + (1 - y) @ torch.log(1 - torch.where(torch.sigmoid(Phi @ weights) > 0.5, 1, 0)) ) 


def find_MAP(Phi, y):
    """
    Find the MAP estimate of the log_joint method.
    
    INPUT:
    Phi   : (N, D) tensor of input features, where N is the number of 
            observations and D is the number of features
    y     : (N,) vector of outputs (targets). Should be a `torch.FloatTensor`
            containing zeros and ones


    OUTPUT:
    w      : (D,) vector of optimized weights
    losses : list of losses at each iteration of the optimization algorithm.
             Should be a list of scalars, which can be plotted afterward to
             diagnose convergence.
    """

    weights = torch.ones(size=(Phi.shape[-1],)).requires_grad_(True)
    #weights = torch.tensor( weights_dist.sample((Phi.shape[-1],)), requires_grad=True )
    
    losses = []
    
    epochs = 100
    optimizer = torch.optim.SGD((weights, Phi, y), lr=0.005, momentum=0.3)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        # forward pass
        loss = -log_joint(Phi, y, weights)
        # backward pass
        loss.backward()
        optimizer.step()            
        
        losses.append( loss.item())
    
    return weights.detach(), losses


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = torch.load("data.pt")

    def features_simple(X):
        return torch.concat((torch.ones_like(X[:,:1]), X), -1)

    def features_quadratic(X):
        interactions = X.prod(-1, keepdim=True)
        return torch.concat((torch.ones_like(X[:,:1]), 
                            X, X.pow(2), interactions), -1)

    print("Dimension of Phi, `features_simple`:", features_simple(X_train).shape)
    print("Dimension of Phi, `features_quadratic`:", features_quadratic(X_train).shape)

    w_MAP_simple, losses = find_MAP(features_simple(X_train), y_train)
    plt.plot(losses);
    plt.xlabel("Iteration")
    plt.ylabel("Loss");
    plt.show()