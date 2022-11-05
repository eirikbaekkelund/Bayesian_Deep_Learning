import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist


measurements = torch.FloatTensor([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])

def algo_parameters():
    """ TODO: set these to appropriate values:
    
    OUTPUT:
    N_samples : total number of MCMC steps
    N_burnin  : number of initial steps to discard
    """   
    N_samples = 800
    N_burnin = int(N_samples/5)
    return N_samples, N_burnin

def log_prior_alpha_beta(alpha, beta):
    """
    Define a prior distribution on alpha, beta, and return its log probability
    
    INPUT:
    alpha : scalar, standard deviation of Gaussian distribution on mu
    beta  : scalar, rate of exponential distribution on sigma_i

    OUTPUT:
    log_prob : scalar, `log p(alpha, beta)`
    
    """
    print("P(alpha) = ", dist.Normal(50, 10, validate_args=False).log_prob(alpha))
    print("P(beta) = ", dist.Normal(0.5, 1, validate_args=False).log_prob(beta))
    return dist.Normal(50, 20, validate_args=False).log_prob(alpha) + dist.Normal(0.5, 5, validate_args=False).log_prob(beta)

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

def mcmc_step_hyperparams(mu, sigma, alpha, beta):
    """
    Run an MCMC step on alpha and beta
    
    INPUT:
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0
    alpha : scalar, standard deviation of Gaussian distribution on mu
    beta  : scalar, rate of exponential distribution on sigma_i

    OUTPUT:
    alpha    : the next value of alpha in the MCMC chain
    beta     : the next value of beta in the MCMC chain
    accepted : a boolean value, indicating whether the proposal was accepted
    
    """
    assert(sigma.shape == (7,))
    assert(mu.shape == ())
    accepted = False

    try:
        proposal_alpha = dist.Normal(loc=alpha, scale=1, validate_args=False).sample()
        proposal_beta = dist.Uniform(0.1, 5, validate_args=False).sample()

        log_prior_new = log_prior_alpha_beta(proposal_alpha, proposal_beta)
        log_joint_new = log_joint(mu, sigma, proposal_alpha, proposal_beta)
        p_new = log_prior_new + log_joint_new
    
    except ValueError:
        return alpha, beta, accepted
    
    # print("Alpha: ", proposal_alpha)
    # print("Beta: ", proposal_beta)
    # print("Prior: ", log_prior_new)
    # print("Joint: ", log_joint_new)
    
    
    log_joint_old = log_joint(mu, sigma, alpha, beta)
    log_prior_old = log_prior_alpha_beta(alpha, beta)

    p_old = log_joint_old + log_prior_old

    if (p_new - p_old) > torch.rand(1).log().item():
        accepted = True
        return proposal_alpha, proposal_beta, accepted
    
    else:
        return alpha, beta, accepted

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
                                      covariance_matrix=torch.eye(sigma.shape[0]))
    q_mu = dist.Normal(loc=mu, scale=1, validate_args=False)
    
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

def run_mcmc_bonus(N_iters, mu_init, sigma_init, alpha_init, beta_init):
    """ Run an MCMC algorithm for a fixed number of iterations.
    
    This also runs MCMC on "hyperparameters" alpha and beta.
    
    """
    
    mu_chain = [mu_init]
    sigma_chain = [sigma_init]
    alpha_chain = [alpha_init]
    beta_chain = [beta_init]
    for _ in range(N_iters):
        alpha, beta, accepted = mcmc_step_hyperparams(mu_chain[-1], sigma_chain[-1], alpha_chain[-1], beta_chain[-1])
        alpha_chain.append(alpha)
        beta_chain.append(beta)

        mu, sigma, accepted = mcmc_step(mu_chain[-1], sigma_chain[-1], alpha_chain[-1], beta_chain[-1])
        mu_chain.append(mu)
        sigma_chain.append(sigma)
    
    return torch.stack(mu_chain), torch.stack(sigma_chain), torch.stack(alpha_chain), torch.stack(beta_chain)

def get_weights_and_samples(mu_chain, sigma_chain, N_burnin):
    mu_chain = mu_chain[N_burnin:]
    sigma_chain = sigma_chain[N_burnin:,:]
   
    proposal_mu, proposal_sigma = get_mcmc_proposal(mu_chain[-1], sigma_chain[-1,:])
    
    samples_mu = proposal_mu.sample((len(mu_chain),))
    samples_sigma = proposal_sigma.sample((len(sigma_chain),))
    
    log_weights = torch.FloatTensor([ log_joint(mu, sig) - 
                                (proposal_mu.log_prob(mu) + proposal_sigma.log_prob(sig)) 
                                for mu, sig in zip(samples_mu, samples_sigma)] )
    log_norm_weights = log_weights - torch.logsumexp(log_weights, dim=-1)
    
    weights = log_norm_weights.exp()

    return weights, samples_mu

def estimate_E_mu(mu_chain, sigma_chain, N_burnin):
    """ Estimate E[mu] 
    
    INPUTS:
    mu_chain    : sequence of MCMC samples of mu
    sigma_chain : sequence of MCMC samples of sigma 
    N_burnin    : number of initial MCMC samples to discard as burnin 
    
    OUTPUTS:
    mu : expected value of mu (scalar)
    """

    weights, samples = get_weights_and_samples(mu_chain, sigma_chain, N_burnin)

    return weights @ samples
    
def estimate_pr_mu_lt_9(mu_chain, sigma_chain, N_burnin):
    """ Estimate the posterior probability that mu is less than 9, i.e. Pr(mu < 9) 
    
    INPUTS:
    mu_chain    : sequence of MCMC samples of mu
    sigma_chain : sequence of MCMC samples of sigma 
    N_burnin    : number of initial MCMC samples to discard as burnin 
    
    OUTPUTS:
    estimate : estimate of Pr(mu < 9), a scalar
    """
    weights, samples = get_weights_and_samples(mu_chain, sigma_chain, N_burnin)
    
    return weights @ (samples < 9).float()



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
    
    # /TODO add plot of posterior
    
    print("acceptance rate:", accepted, "\n")
    print("E[mu] = %0.4f" % estimate_E_mu(mu_chain, sigma_chain, N_burnin), '\n')
    print("Pr(mu < 9) = %0.4f" % estimate_pr_mu_lt_9(mu_chain, sigma_chain, N_burnin), '\n')
 
    plt.plot(mu_chain);
    plt.xlabel("MCMC iteration");
    plt.ylabel("$\mu$")
    plt.figure();
    plt.hist(mu_chain[N_burnin:].numpy(), bins=20);
    plt.xlabel("$\mu$")
    plt.ylabel("Counts");
    plt.show()

 
    plt.figure(figsize=(12,4));
    plt.plot(sigma_chain)
    plt.legend(range(1,8));
    plt.xlabel("MCMC iteration")
    plt.ylabel("$\sigma_i$");
    plt.show()


    plt.boxplot(sigma_chain[N_burnin:].T, positions=np.arange(1, 8));
    plt.xlabel("Which scientist")
    plt.ylabel("Estimated measurement std $\sigma_i$");

    plt.show()

    new_mu_chain, new_sigma_chain, alpha_chain, beta_chain = run_mcmc_bonus(N_samples, mu_chain[-1], sigma_chain[-1], torch.tensor(50.0), torch.tensor(0.5))

    plt.plot(mu_chain)
    plt.plot(alpha_chain);
    plt.plot(beta_chain);
    plt.legend(['mu', 'alpha', 'beta']);
    plt.xlabel("Iteration");
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(alpha_chain[N_burnin:].numpy(), bins=20, density=True);
    plt.xlabel("$\\alpha$")
    plt.ylabel("$p(\\alpha)$")
    plt.show()
    
    plt.subplot(122)
    plt.hist(beta_chain[N_burnin:].numpy(), bins=20, density=True);
    plt.xlabel("$\\beta$")
    plt.ylabel("$p(\\beta)$");
    plt.show()

    plt.hist(new_mu_chain[N_burnin:].numpy(), bins=20, density=True);
    plt.xlabel("$\mu$")
    plt.ylabel("$p(\mu)$");
    plt.show()

    plt.boxplot(new_sigma_chain[N_burnin:].T, positions=np.arange(1, 8));
    plt.xlabel("Which scientist")
    plt.ylabel("Estimated measurement std $\sigma_i$");
    plt.show()