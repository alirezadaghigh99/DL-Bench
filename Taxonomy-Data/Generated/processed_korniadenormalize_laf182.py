import torch

def denormalize_laf(LAF, images):
    """
    De-normalize LAFs from scale to image scale.

    Args:
        LAF: torch.Tensor of shape (B, N, 2, 3)
        images: torch.Tensor of shape (B, CH, H, W)

    Returns:
        torch.Tensor: the denormalized LAF of shape (B, N, 2, 3), scale in pixels
    """
    B, N, _, _ = LAF.size()
    _, _, H, W = images.size()
    MIN_SIZE = min(H - 1, W - 1)

    # Create a scaling matrix
    scaling_matrix = torch.tensor([
        [MIN_SIZE, 0, W - 1],
        [0, MIN_SIZE, W - 1]
    ], dtype=LAF.dtype, device=LAF.device)

    # Apply the scaling to each LAF
    denormalized_LAF = LAF.clone()
    for b in range(B):
        for n in range(N):
            denormalized_LAF[b, n, :, :] = torch.mm(LAF[b, n, :, :], scaling_matrix)

    return denormalized_LAF

