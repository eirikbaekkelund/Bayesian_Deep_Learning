import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

measurements = torch.FloatTensor([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])


def log_joint(mu, sigma, alpha=50, beta=0.5):
    """
    INPUT:
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0
    alpha : scalar, standard deviation of Gaussian prior on mu. Default to 50
    beta  : scalar, rate of exponential prior on sigma_i. Default to 0.5

    OUTPUT:
    log_joint: the log probability log p(mu, sigma, x | alpha, beta), scalar
    
    NOTE: For inputs where sigma <= 0, please return negative infinity!

    """
    assert mu.ndim == 0
    assert sigma.shape == (7,)
    
    if torch.any(sigma <= 0):
        return -np.inf

    log_prior_mu = dist.Normal(loc=0, scale=alpha**2, validate_args=False).log_prob(mu)
    log_prior_sigma = dist.Exponential(rate=beta).log_prob(sigma).sum(0)
    
    log_likelihood = torch.sum(torch.FloatTensor([dist.Normal(loc=mu, scale=sig).log_prob(xi) for sig, xi in zip(sigma, measurements)]) )
  

    return log_prior_mu + log_prior_sigma + log_likelihood

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

    q_sigma = dist.MultivariateNormal(loc=sigma, 
                                      covariance_matrix= 0.3*torch.eye(sigma.shape[0]))
    q_mu = dist.Normal(loc=mu, scale=0.5)
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

    if (p_new - p_old) > torch.rand(1).log().item():
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
    N_burnin = int(N_samples/10)
    return N_samples, N_burnin


if __name__ == "__main__":

    sigma=dist.Exponential(0.5).sample((7,))
    mu = dist.Normal(0, 50**2).sample()
    x = torch.FloatTensor([dist.Normal(loc=mu, scale=sigma[i]).sample() for i in range(len(sigma))])  

    # plt.plot(np.linspace(-7500, 7500, 1000), [log_joint(mu, sigma) for mu in np.linspace(-7500,7500,1000)], label='p($\mu, X, \sigma, \alpha, \beta $)')
    # plt.axvline(x=mu, color='black', linestyle='--', label='$\hat{\mu}$')
    # plt.legend()
    # plt.show()

    q_mu, q_sigma = get_mcmc_proposal(torch.tensor(9.0), torch.ones(7))
    assert isinstance(q_mu, dist.Distribution)
    assert isinstance(q_sigma, dist.Distribution)
    assert q_mu.sample().shape == ()
    assert q_sigma.sample().shape == (7,)
    del q_mu, q_sigma

    measurements = torch.FloatTensor([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])

    mu_init = measurements.mean()
    sigma_init = torch.ones(7)

    N_samples, N_burnin = algo_parameters()

    mu_chain, sigma_chain, accepted = run_mcmc(N_samples, mu_init, sigma_init)
    print("acceptance rate:", accepted)

    # PLOT MEAN
    # plt.plot(mu_chain);
    # plt.xlabel("MCMC iteration");
    # plt.ylabel("$\mu$")
    # plt.figure();
    # plt.hist(mu_chain[N_burnin:].numpy(), bins=20);
    # plt.xlabel("$\mu$")
    # plt.ylabel("Counts");
    # plt.show()

    # PLOT VARIANCE
    # plt.figure(figsize=(12,4));
    # plt.plot(sigma_chain)
    # plt.legend(range(1,8));
    # plt.xlabel("MCMC iteration")
    # plt.ylabel("$\sigma_i$");
    # plt.show()


    # plt.boxplot(sigma_chain[N_burnin:].T, positions=np.arange(1, 8));
    # plt.xlabel("Which scientist")
    # plt.ylabel("Estimated measurement std $\sigma_i$");