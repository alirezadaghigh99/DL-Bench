import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.0")
    
    # Calculate the normalization constant
    squared_offsets = offsets ** 2
    sum_squared_offsets = torch.sum(squared_offsets, dim=-1)
    normalization_constant = torch.exp(-sum_squared_offsets / (2 * sigma ** 2))
    
    # Add epsilon to ensure gradient flow
    epsilon = 0.05
    normalization_constant += epsilon
    
    # Calculate the normalized value
    normalized_value = (1 + epsilon) / normalization_constant
    
    return normalized_value