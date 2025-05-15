import torch

def _unsafe_standard_stable(alpha, beta, aux_uniform, aux_exponential):
    # This function should implement the standard stable distribution transformation
    # for the general case. This is a placeholder for the actual implementation.
    # For demonstration purposes, let's assume it returns a tensor of the same shape.
    # In practice, you would replace this with the actual transformation logic.
    return torch.zeros_like(aux_uniform)  # Placeholder implementation

def _standard_stable(alpha, beta, aux_uniform, aux_exponential, coords):
    RADIUS = 1e-2  # Define a small radius for checking if alpha is near 1

    if coords not in ["S", "S0"]:
        raise ValueError(f"Unknown coords: {coords}")

    # Convert alpha and beta to tensors if they are not already
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=aux_uniform.dtype, device=aux_uniform.device)
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=aux_uniform.dtype, device=aux_uniform.device)

    # Handle the case where alpha is near 1
    if torch.abs(alpha - 1) < RADIUS:
        # Interpolation workaround for alpha near 1
        alpha1 = torch.tensor(1.0 + RADIUS, dtype=aux_uniform.dtype, device=aux_uniform.device)
        alpha2 = torch.tensor(1.0 - RADIUS, dtype=aux_uniform.dtype, device=aux_uniform.device)
        
        # Compute the two points
        point1 = _unsafe_standard_stable(alpha1, beta, aux_uniform, aux_exponential)
        point2 = _unsafe_standard_stable(alpha2, beta, aux_uniform, aux_exponential)
        
        # Interpolate between the two points
        weight = (alpha - alpha2) / (alpha1 - alpha2)
        result = weight * point1 + (1 - weight) * point2
    else:
        # Directly call the unsafe standard stable function
        result = _unsafe_standard_stable(alpha, beta, aux_uniform, aux_exponential)

    # Handle the case where coords is "S"
    if coords == "S":
        # Apply the correction for "S" coordinates
        correction = beta * torch.tan(torch.tensor(torch.pi / 2, dtype=aux_uniform.dtype, device=aux_uniform.device))
        result += correction

    return result

