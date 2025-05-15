import torch

def _normalize_and_compose_all_layers(background_color, splatted_colors_per_occlusion_layer, splatted_weights_per_occlusion_layer):
    # Ensure the background color is a tensor
    background_color = torch.tensor(background_color, dtype=torch.float32)
    
    # Unpack the dimensions of the input tensors
    N, H, W, _, _ = splatted_colors_per_occlusion_layer.shape
    
    # Initialize the output colors with the background color
    output_colors = torch.zeros((N, H, W, 4), dtype=torch.float32)
    output_colors[..., :3] = background_color
    
    # Iterate over each occlusion layer (foreground, surface, background)
    for i in range(3):
        # Extract the colors and weights for the current layer
        colors = splatted_colors_per_occlusion_layer[..., i, :]
        weights = splatted_weights_per_occlusion_layer[..., i]
        
        # Normalize the colors by their weights
        normalized_colors = torch.where(weights > 0, colors / weights, colors)
        
        # Extract the alpha channel (assuming it's the last channel in RGBA)
        alpha = normalized_colors[..., 3:4]
        
        # Perform alpha compositing
        output_colors[..., :3] = alpha * normalized_colors[..., :3] + (1 - alpha) * output_colors[..., :3]
        output_colors[..., 3:4] = alpha + (1 - alpha) * output_colors[..., 3:4]
    
    return output_colors

