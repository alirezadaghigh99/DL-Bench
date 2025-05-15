import torch
import torch.nn.functional as F

def _compute_occlusion_layers(q_depth):
    """
    Compute occlusion layers for each splatting pixel.

    Args:
        q_depth (torch.Tensor): Input depth tensor of shape (N, H, W).

    Returns:
        torch.Tensor: Occlusion layers tensor of shape (N, H, W, 9).
    """
    N, H, W = q_depth.shape
    occlusion_layers = torch.zeros((N, H, W, 9), device=q_depth.device)

    # Define the 9 directions: center, N, S, E, W, NE, NW, SE, SW
    directions = [
        (0, 0),  # center
        (-1, 0), # north
        (1, 0),  # south
        (0, 1),  # east
        (0, -1), # west
        (-1, 1), # northeast
        (-1, -1),# northwest
        (1, 1),  # southeast
        (1, -1)  # southwest
    ]

    for i, (dy, dx) in enumerate(directions):
        # Shift the depth tensor according to the direction
        shifted_depth = F.pad(q_depth, (1, 1, 1, 1), mode='replicate')
        shifted_depth = shifted_depth[:, 1+dy:H+1+dy, 1+dx:W+1+dx]

        # Compare the original depth with the shifted depth
        same_surface = (q_depth == shifted_depth).float()
        background = (q_depth > shifted_depth).float()
        foreground = (q_depth < shifted_depth).float()

        # Assign values to the occlusion layers
        occlusion_layers[..., i] = same_surface + 2 * background + 3 * foreground

    return occlusion_layers

