import torch
import torch.distributions as dist
import numpy as np
import math
import matplotlib.pyplot as plt

class MonteCarlo:
    
    def __init__(self, n_samples, n_observed):
        self.distribution = dist.Uniform(0,1)
        self.samples, self.observed_toss = self.generate_samples(n_samples, 
                                                                 n_observed,
                                                                 distribution=dist.Uniform(0,1))
        
        self.weights = self.initialize_weights(self.samples, distribution=dist.Uniform(0,1))
        
        print(f'μ = {(self.observed_toss.sum() / len(self.observed_toss)).item()}')

    def log_joint_bernoulli(self,mu, X):
        """
        Evaluate the log joint probability distribution:
        P(mu, X) = P(mu) * P(X|mu)

        Parameters:
        mu - the provided mu for our Bernoulli(mu) distribution to sample from
        X - observed data to provide to our joint distribution

        Returns:
        log P(mu, X) = log P(mu) + log P(X|mu)
        That is, the joint log probability
        """
        # P(mu)
        log_prior = dist.Uniform(0, 1, validate_args=False).log_prob(mu)
        
        # P(X|mu) assuming independence
        log_posterior = dist.Bernoulli(mu).log_prob(X).sum(0)
        # log[ P(mu)P(X|mu) ]
        return log_prior + log_posterior
    
    def generate_samples(self, n_samples, n_observed, distribution):
            """
            Generate synthetic data for observed samples of a coin toss
            where 1 = heads and 0 = tails & samples from a 

            Parameters:
            n_samples - number of samples to draw from distribution
            distribution - probability distribution to draw samples from

            Returns:
            proposal_samples - list with proposal samples from a uniform distribution
            observed_samples - list with observed coin tosses
            """
            rng = np.random.default_rng(seed=12345)
            self.observed_toss = torch.FloatTensor([1 if x > 0.5 else 0 for x in rng.random(size=n_observed)])
        
            return distribution.sample((n_samples,)), self.observed_toss

    def initialize_weights(self, samples, distribution, normalized=True):
        """
        Generate normalized/unnormalized weights for importance sampling
        
        Parameters: 
        samples - samples generated from the proposal distribution
        X - observed data from the proposal distribution

        Returns:
        weights - weights to use in importance sampling
        """
        # w(mu) = p(x|mu)p(mu) / q(mu) ; unnormalized
        weights = self.log_joint_bernoulli(self.samples, self.observed_toss[:,None]) - distribution.log_prob(samples)

        if normalized == True:
            # normalize weights
            weights = weights - torch.logsumexp(weights, dim=-1)
            # apply exponent to remove log on weights
            weights = weights.exp()
            
            assert(0.99 <= weights.sum().item() <= 1.01)

        return weights

    def expected_probability_heads(self):
        """
        Generate proposal samples from a proposal distribution q(mu).
        Then computes the weights for the proposal samples, and computes
        the probability of heads from a Bernoulli given observed samples
        from our sampling distribution.
        """
        # E_{p(mu|X)} [f(x)] = W @ f = P(heads)
        return self.weights @ self.samples

    def compute_squared_difference(self):
        """
        Computes the squared difference between empirical probability of heads
        from observed data and P(next toss is heads).
        """
        # E_{p(mu|X)} [f(x)] = W @ f = P(heads)
        P_heads = self.weights @ self.samples
        print(f"P(H) = {P_heads}")
        empirical_probability = self.observed_toss.sum() / len(self.observed_toss)
        
        return (P_heads - empirical_probability)**2
        
    def percentage_greater_than_prob(self, percentage):
        """
        Calculates the probability that >p% of the samples are heads
        """
        # P(X > p)
        return self.weights @ (self.samples > percentage).float()

    def percentage_less_than_prob(self, percentage):
        """
        Calculates the probability that >p% of the samples are heads
        """
        # P(X < p)
        return self.weights @ (self.samples < percentage).float()
    
    def markov_chain_step(self, current_mu, observed_data, proposal_std):
        """
        Markov Chain Monte Carlo step-wise function.
        Accepts a new value 
        """
        proposal_Q = dist.Normal(loc=current_mu, 
                                 scale=proposal_std)
        sample_x_prime = proposal_Q.sample()


        try:
            prob_sample_joint = self.log_joint_bernoulli(mu=sample_x_prime,
                                                         X=observed_data)
        except ValueError:
            prob_sample_joint = -torch.inf
        
        prob_current_joint = self.log_joint_bernoulli(mu=current_mu,
                                                      X=observed_data)
        
        if prob_sample_joint - prob_current_joint > torch.rand(1).log().item():
            return sample_x_prime, 1
        else:
            return current_mu, 0
    
    def run_markov_chain(self, init, num_steps, proposal_std):
        """
        Computes a sequence of samples and the acceptance rate

        Parameters:
        init - guessed initial value to start at
        num_steps - number of times to run MCMC
        proposal_std - suggested variance for the normal distribution

        Returns:
        mu_chain - array of suggested values for mu in Normal from MCMC
        acceptance_rate - proportion of suggested samples accepted
        """

        self.mu_chain = [torch.scalar_tensor(init)]
        n_accepted = 0

        for _ in range(num_steps):
            mu, accept_bool = self.markov_chain_step(self.mu_chain[-1], 
                                                self.observed_toss,
                                                proposal_std=proposal_std)
            self.mu_chain.append(mu)
            n_accepted += accept_bool
        
        self.acceptance_rate = n_accepted / num_steps
        
        return self.mu_chain, self.acceptance_rate
    
    def plot_chain_steps(self):
        
        plt.plot(self.mu_chain, label='$\mu$ samples')
        plt.title(f"MCMC Acceptance Rate: {self.acceptance_rate}")
        plt.ylim(0, 1);
        plt.xlabel("MCMC iteration")
        plt.ylabel('$\mu$');
        plt.show()
    
    def plot_posterior(self):
        domain = np.linspace(0,1, 300)
        posterior = self.log_joint_bernoulli(mu=torch.FloatTensor(domain), 
                                             X=self.observed_toss[:,None]).exp()

        plt.plot(domain, posterior, linestyle='--', label='$P(\mu | X)$')
        plt.axvline(x=np.mean(self.mu_chain), color='red', label='MCMC $\mu$')
        plt.axvline(x=np.mean(self.observed_toss.numpy()), color= 'black', label='Observed $\mu$')
        plt.xlim(0, 1)
        plt.xlabel("$\mu$")
        plt.ylabel("$p(\mu | x_1,\dots,x_N)$")
        plt.title("MCMC histogram of posterior");
        plt.legend()
        plt.show()

        

        
        
        
        
