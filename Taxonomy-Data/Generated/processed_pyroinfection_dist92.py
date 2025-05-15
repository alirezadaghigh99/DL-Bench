import numpy as np
from scipy.stats import poisson, nbinom, binom, betabinom

def infection_dist(individual_rate, num_infectious, num_susceptible, population, concentration, overdispersion):
    # Calculate the basic reproduction number (R0) for the current scenario
    R0 = individual_rate * num_infectious / population
    
    # Calculate the expected number of new infections
    expected_infections = R0 * num_susceptible
    
    # Determine the appropriate distribution based on the parameters
    if population > 1000 and concentration > 1:
        # Use Poisson distribution for large populations and high concentration
        return poisson(mu=expected_infections)
    
    elif overdispersion > 1:
        # Use Negative Binomial distribution for overdispersed data
        # The negative binomial distribution is parameterized by the number of successes (n) and the probability of success (p)
        # We need to convert the mean and overdispersion to these parameters
        p = concentration / (concentration + expected_infections)
        n = concentration
        return nbinom(n=n, p=p)
    
    elif population < 1000:
        # Use Binomial distribution for small populations
        # The binomial distribution is parameterized by the number of trials (n) and the probability of success (p)
        p = expected_infections / num_susceptible
        return binom(n=num_susceptible, p=p)
    
    else:
        # Use Beta-Binomial distribution for scenarios with variability in individual susceptibility
        # The beta-binomial distribution is parameterized by the number of trials (n), alpha, and beta
        alpha = concentration * expected_infections / num_susceptible
        beta = concentration * (1 - expected_infections / num_susceptible)
        return betabinom(n=num_susceptible, a=alpha, b=beta)

