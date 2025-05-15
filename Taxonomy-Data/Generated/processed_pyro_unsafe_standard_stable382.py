import torch

def _unsafe_standard_stable(alpha, beta, V, W, coords):
    # Ensure V and W have the same shape
    assert V.shape == W.shape, "V and W must have the same shape"
    
    # Check if alpha is close to 1, which is not supported
    if torch.isclose(torch.tensor(alpha), torch.tensor(1.0)):
        raise ValueError("Alpha close to 1 is not supported")
    
    # Define a small epsilon for numerical stability
    eps = 1e-10
    
    # Calculate pi/2 and pi*alpha/2 for reuse
    pi_over_2 = torch.tensor(torch.pi / 2)
    pi_alpha_over_2 = torch.tensor(torch.pi * alpha / 2)
    
    # Calculate the angle and the part of the formula that depends on V
    angle = pi_over_2 * (2 * V - 1)
    part1 = torch.tan(angle)
    
    # Calculate the part of the formula that depends on W
    part2 = (1 - alpha) * torch.log(W + eps)
    
    # Calculate the stable random variable based on the coords
    if coords == "S0":
        # S0 coordinate system
        factor = (1 + beta**2 * part1**2)**(1 / (2 * alpha))
        result = factor * (torch.sin(alpha * angle) / (torch.cos(angle)**(1 / alpha))) * \
                 ((torch.cos((1 - alpha) * angle) / W)**((1 - alpha) / alpha))
    elif coords == "S":
        # S coordinate system
        result = (torch.sin(alpha * angle) / (torch.cos(angle)**(1 / alpha))) * \
                 ((torch.cos((1 - alpha) * angle) / W)**((1 - alpha) / alpha))
    else:
        raise ValueError(f"Unknown coords: {coords}")
    
    # Replace NaN values with zeros
    result = torch.nan_to_num(result, nan=0.0)
    
    return result

