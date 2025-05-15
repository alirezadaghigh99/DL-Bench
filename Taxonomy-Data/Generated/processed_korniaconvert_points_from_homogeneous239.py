import torch

def convert_points_from_homogeneous(points, eps=1e-10):
    # Check if the input is a tensor
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input must be a tensor.")
    
    # Check if the input tensor has at least two dimensions
    if points.dim() < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    
    # Get the shape of the input tensor
    B, N, D = points.shape
    
    # Extract the last coordinate for each point
    last_coord = points[..., -1:]
    
    # Avoid division by zero by adding eps to the last coordinate
    last_coord = last_coord.clamp(min=eps)
    
    # Divide each point by its last coordinate to convert to Euclidean space
    euclidean_points = points[..., :-1] / last_coord
    
    return euclidean_points

