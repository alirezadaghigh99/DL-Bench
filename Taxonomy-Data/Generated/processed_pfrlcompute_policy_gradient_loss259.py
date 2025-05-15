import numpy as np

def compute_policy_gradient_loss(action, advantage, action_distrib, action_distrib_mu, action_value, v, truncation_threshold=None):
    """
    Computes the policy gradient loss with off-policy bias correction.

    Parameters:
    - action: The action taken.
    - advantage: The advantage of taking that action.
    - action_distrib: The distribution of actions (target policy).
    - action_distrib_mu: The distribution of actions from the behavior policy.
    - action_value: The value of the action taken.
    - v: The value function.
    - truncation_threshold: Optional threshold for truncating the off-policy policy gradient term.

    Returns:
    - The policy gradient loss as a scalar value.
    """
    # Compute the probability of the action under the target policy
    pi_a = action_distrib[action]
    
    # Compute the probability of the action under the behavior policy
    mu_a = action_distrib_mu[action]
    
    # Compute the importance sampling ratio
    rho = pi_a / (mu_a + 1e-10)  # Add a small constant to avoid division by zero
    
    # Apply truncation if a threshold is provided
    if truncation_threshold is not None:
        rho = min(rho, truncation_threshold)
    
    # Compute the off-policy corrected advantage
    corrected_advantage = rho * (advantage + action_value - v)
    
    # Compute the policy gradient loss
    policy_gradient_loss = -corrected_advantage
    
    return policy_gradient_loss

