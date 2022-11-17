import sys
sys.settrace
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

measurements = torch.FloatTensor([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])


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

def get_mcmc_proposal(mu, sigma):
    """
    INPUT:
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0

    OUTPUT:
    q_mu    : instance of Distribution class, that defines a proposal for mu
    q_sigma : instance of Distribution class, that defines a proposal for sigma
    """
    assert sigma.shape == (7,)
    assert( torch.any(sigma > 0))
    assert(mu.shape == ())
    print(mu.shape, sigma.shape)   

    q_sigma = dist.MultivariateNormal(loc=sigma, 
                                      covariance_matrix= torch.eye(sigma.shape[0]))
    q_mu = dist.Normal(loc=mu, scale=torch.mean(sigma))
    return q_mu, q_sigma

def mcmc_step(mu, sigma, alpha=50, beta=0.5):
    """
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0
    alpha : scalar, standard deviation of Gaussian prior on mu. Default to 50
    beta  : scalar, rate of exponential prior on sigma_i. Default to 0.5

    OUTPUT:
    mu       : the next value of mu in the MCMC chain
    sigma    : the next value of sigma in the MCMC chain
    accepted : a boolean value, indicating whether the proposal was accepted

    """
    
    accepted = False
    q_mu, q_sigma = get_mcmc_proposal(mu, sigma)
    
    new_mu = q_mu.sample()
    new_sigma = q_sigma.sample()

    p_new = log_joint(new_mu, new_sigma)
    p_old = log_joint(mu, sigma)

    if p_new - p_old > torch.rand(1).log().item():
        accepted = True
        return new_mu, new_sigma, accepted
    else:
        return mu, sigma, accepted

def run_mcmc(N_iters, mu_init, sigma_init):
    """ Run an MCMC algorithm for a fixed number of iterations """
    
    mu_chain = [mu_init]
    sigma_chain = [sigma_init]
    N_accepted = 0
    for _ in range(N_iters):
        mu, sigma, accepted = mcmc_step(mu_chain[-1], sigma_chain[-1])
        mu_chain.append(mu)
        sigma_chain.append(sigma)
        N_accepted += accepted
    
    return torch.stack(mu_chain), torch.stack(sigma_chain), N_accepted / N_iters

def algo_parameters():
    """ TODO: set these to appropriate values:
    
    OUTPUT:
    N_samples : total number of MCMC steps
    N_burnin  : number of initial steps to discard
    """   
    N_samples = 1000
    N_burnin = int(N_samples/4)
    return N_samples, N_burnin


if __name__ == "__main__":
    print(os.path.abspath('data.pt'))
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
    q_mu, q_sigma = get_mcmc_proposal(torch.tensor(9.0), torch.ones(7))
    assert isinstance(q_mu, dist.Distribution)
    assert isinstance(q_sigma, dist.Distribution)
    assert q_mu.sample().shape == ()
    assert q_sigma.sample().shape == (7,)


    mu_init = measurements.mean()
    sigma_init = torch.ones(7)

    N_samples, N_burnin = algo_parameters()

    mu_chain, sigma_chain, accepted = run_mcmc(N_samples, mu_init, sigma_init)
    print("acceptance rate:", accepted)
    plt.plot(mu_chain);
    plt.xlabel("MCMC iteration");
    plt.ylabel("$\mu$")
    plt.figure();
    plt.hist(mu_chain[N_burnin:].numpy(), bins=20);
    plt.xlabel("$\mu$")
    plt.ylabel("Counts");

    plt.figure(figsize=(12,4));
    plt.plot(sigma_chain)
    plt.legend(range(1,8));
    plt.xlabel("MCMC iteration")
    plt.ylabel("$\sigma_i$");