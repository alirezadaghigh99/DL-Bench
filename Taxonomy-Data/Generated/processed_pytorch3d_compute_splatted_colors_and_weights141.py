import torch

def _compute_splatted_colors_and_weights(occlusion_layers, splat_colors_and_weights):
    """
    Accumulates splatted colors in background, surface, and foreground occlusion buffers.

    Args:
        occlusion_layers (torch.Tensor): A tensor of shape (N, H, W, 9).
        splat_colors_and_weights (torch.Tensor): A tensor of shape (N, H, W, K, 9, 5).

    Returns:
        tuple: A tuple containing:
            - splatted_colors (torch.Tensor): A tensor of shape (N, H, W, 4, 3).
            - splatted_weights (torch.Tensor): A tensor of shape (N, H, W, 1, 3).
    """
    N, H, W, K, _, _ = splat_colors_and_weights.shape

    # Initialize the output tensors
    splatted_colors = torch.zeros((N, H, W, 4, 3), dtype=splat_colors_and_weights.dtype, device=splat_colors_and_weights.device)
    splatted_weights = torch.zeros((N, H, W, 1, 3), dtype=splat_colors_and_weights.dtype, device=splat_colors_and_weights.device)

    # Iterate over each occlusion layer (background, surface, foreground)
    for i in range(3):
        # Extract the weights and colors for the current occlusion layer
        weights = splat_colors_and_weights[..., i, 4]  # Shape: (N, H, W, K)
        colors = splat_colors_and_weights[..., i, :3]  # Shape: (N, H, W, K, 3)

        # Accumulate the weighted colors
        weighted_colors = colors * weights.unsqueeze(-1)  # Shape: (N, H, W, K, 3)
        splatted_colors[..., i, :] = weighted_colors.sum(dim=3)  # Sum over K

        # Accumulate the weights
        splatted_weights[..., 0, i] = weights.sum(dim=3)  # Sum over K

    # Handle the alpha channel for splatted_colors
    splatted_colors[..., 3, :] = splatted_weights[..., 0, :].unsqueeze(-1)  # Copy weights to alpha channel

    return splatted_colors, splatted_weights