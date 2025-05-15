import torch

def compute_policy_gradient_full_correction(action_distrib, action_distrib_mu, action_value, v, truncation_threshold):
    # Assertions to ensure correct input types and values
    assert isinstance(action_distrib, torch.distributions.Categorical), "action_distrib must be a torch.distributions.Categorical"
    assert isinstance(action_distrib_mu, torch.distributions.Categorical), "action_distrib_mu must be a torch.distributions.Categorical"
    assert isinstance(action_value, torch.Tensor), "action_value must be a torch.Tensor"
    assert isinstance(v, (int, float)), "v must be a scalar (int or float)"
    assert isinstance(truncation_threshold, (int, float)), "truncation_threshold must be a scalar (int or float)"
    assert action_value.dim() == 1, "action_value must be a 1D tensor"
    assert action_distrib.probs.shape == action_distrib_mu.probs.shape == action_value.shape, "Shapes of action_distrib, action_distrib_mu, and action_value must match"

    with torch.no_grad():
        # Compute importance weights
        importance_weights = action_distrib.probs / action_distrib_mu.probs
        
        # Apply truncation
        truncated_weights = torch.clamp(importance_weights, max=truncation_threshold)
    
    # Calculate advantage
    advantage = action_value - v
    
    # Compute corrected policy gradient
    corrected_policy_gradient = truncated_weights * advantage
    
    return corrected_policy_gradient

