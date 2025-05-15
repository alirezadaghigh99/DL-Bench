import torch
import torch.nn.functional as F

def equalize_clahe(input, clip_limit, grid_size, slow_and_differentiable=False):
    # Error handling
    if not isinstance(clip_limit, float):
        raise TypeError("clip_limit must be a float.")
    if not (isinstance(grid_size, tuple) and len(grid_size) == 2 and all(isinstance(x, int) for x in grid_size)):
        raise TypeError("grid_size must be a tuple of two integers.")
    if any(x <= 0 for x in grid_size):
        raise ValueError("All elements of grid_size must be positive.")

    # Ensure input is a float tensor
    input = input.float()

    # Get the shape of the input
    original_shape = input.shape
    batch_dims = original_shape[:-3]
    C, H, W = original_shape[-3:]

    # Calculate tile size
    tile_h = H // grid_size[0]
    tile_w = W // grid_size[1]

    # Pad the image if necessary
    pad_h = (grid_size[0] * tile_h - H) if H % grid_size[0] != 0 else 0
    pad_w = (grid_size[1] * tile_w - W) if W % grid_size[1] != 0 else 0
    if pad_h > 0 or pad_w > 0:
        input = F.pad(input, (0, pad_w, 0, pad_h), mode='reflect')

    # Recompute H and W after padding
    _, H, W = input.shape[-3:]

    # Reshape input to process each tile
    input = input.unfold(-2, tile_h, tile_h).unfold(-1, tile_w, tile_w)
    input = input.contiguous().view(*batch_dims, C, grid_size[0], grid_size[1], tile_h, tile_w)

    # Apply CLAHE to each tile
    def apply_clahe(tile):
        # Compute histogram
        hist = torch.histc(tile, bins=256, min=0.0, max=1.0)
        if clip_limit > 0:
            # Clip histogram
            excess = hist - clip_limit
            excess = torch.clamp(excess, min=0)
            hist = hist - excess
            # Redistribute excess
            hist += excess.sum() / 256
        # Compute CDF
        cdf = hist.cumsum(0)
        cdf = cdf / cdf[-1]  # Normalize
        # Map the tile using the CDF
        tile_flat = tile.view(-1)
        tile_eq = torch.interp(tile_flat, torch.linspace(0, 1, 256), cdf)
        return tile_eq.view(tile.shape)

    # Process each tile
    if slow_and_differentiable:
        # Use a differentiable approach
        output = torch.stack([apply_clahe(input[..., i, j, :, :]) for i in range(grid_size[0]) for j in range(grid_size[1])], dim=-1)
        output = output.view(*batch_dims, C, grid_size[0], grid_size[1], tile_h, tile_w)
    else:
        # Use a faster, non-differentiable approach
        output = torch.zeros_like(input)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                output[..., i, j, :, :] = apply_clahe(input[..., i, j, :, :])

    # Reshape back to original padded size
    output = output.view(*batch_dims, C, H, W)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        output = output[..., :original_shape[-2], :original_shape[-1]]

    return output

