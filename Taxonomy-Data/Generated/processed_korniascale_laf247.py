import torch

def scale_laf(laf, scale_coef):
    """
    Scales the region part of a Local Affine Frame (LAF) by a scale coefficient.

    Parameters:
    - laf (torch.Tensor): A tensor of shape (B, N, 2, 3) representing the LAFs.
    - scale_coef (float or torch.Tensor): The scale coefficient to apply.

    Returns:
    - torch.Tensor: The scaled LAF of the same shape as the input.
    
    Raises:
    - TypeError: If scale_coef is neither a float nor a tensor.
    """
    if not isinstance(scale_coef, (float, torch.Tensor)):
        raise TypeError("scale_coef must be either a float or a torch.Tensor")

    # Ensure scale_coef is a tensor for consistent operations
    if isinstance(scale_coef, float):
        scale_coef = torch.tensor(scale_coef, dtype=laf.dtype, device=laf.device)

    # Extract the region part (first two columns) and the center part (last column)
    region = laf[..., :2]  # Shape: (B, N, 2, 2)
    center = laf[..., 2]   # Shape: (B, N, 2)

    # Scale the region part
    scaled_region = region * scale_coef.unsqueeze(-1).unsqueeze(-1)

    # Reconstruct the scaled LAF
    scaled_laf = torch.cat((scaled_region, center.unsqueeze(-1)), dim=-1)

    return scaled_laf

