import torch

def masked_gather(points, idx):
    # Check if the batch dimensions match
    if points.size(0) != idx.size(0):
        raise ValueError("Batch dimensions of points and idx must match.")
    
    # Replace -1 indices with 0
    idx_replaced = idx.clone()
    idx_replaced[idx_replaced == -1] = 0
    
    # Gather points using the modified indices
    gathered_points = torch.gather(points, 1, idx_replaced.unsqueeze(-1).expand(-1, -1, points.size(-1)))
    
    # Create a mask for the original -1 indices
    mask = (idx == -1).unsqueeze(-1).expand_as(gathered_points)
    
    # Set the gathered values corresponding to the original -1 indices to 0.0
    gathered_points[mask] = 0.0
    
    return gathered_points

