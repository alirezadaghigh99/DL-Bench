import torch

def _compute_splatting_colors_and_weights(pixel_coords_screen, colors, sigma, offsets):
    # Ensure inputs are tensors
    pixel_coords_screen = torch.tensor(pixel_coords_screen)
    colors = torch.tensor(colors)
    offsets = torch.tensor(offsets)

    # Unpack the shape of the input tensors
    N, H, W, K, _ = pixel_coords_screen.shape

    # Prepare the output tensor
    splat_colors_and_weights = torch.zeros((N, H, W, K, 9, 5), dtype=colors.dtype)

    # Compute the Gaussian weights
    for i in range(9):
        # Calculate the offset coordinates for the splatting pixels
        offset_coords = pixel_coords_screen + offsets[i]

        # Compute the squared distance between the center pixel and the splatting pixels
        squared_distances = torch.sum((pixel_coords_screen - offset_coords) ** 2, dim=-1)

        # Compute the Gaussian weights
        weights = torch.exp(-squared_distances / (2 * sigma ** 2))

        # Assign the colors and weights to the output tensor
        splat_colors_and_weights[..., i, :4] = colors
        splat_colors_and_weights[..., i, 4] = weights

    return splat_colors_and_weights

