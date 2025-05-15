import torch

def sample_is_valid_for_homography(points1, points2):
    # Check if the input shapes are equal
    if points1.shape != points2.shape:
        raise ValueError("Input tensors must have the same shape.")
    
    B, N, C = points1.shape
    if N != 4 or C != 2:
        raise ValueError("Input tensors must have shape (B, 4, 2).")
    
    # Convert to homogeneous coordinates
    ones = torch.ones((B, 4, 1), dtype=points1.dtype, device=points1.device)
    points1_h = torch.cat([points1, ones], dim=-1)  # Shape: (B, 4, 3)
    points2_h = torch.cat([points2, ones], dim=-1)  # Shape: (B, 4, 3)
    
    # Function to compute the oriented constraint
    def oriented_constraint(p1, p2):
        # Compute cross product of vectors (p1[1] - p1[0]) x (p1[2] - p1[0])
        cross1 = torch.cross(p1[:, 1] - p1[:, 0], p1[:, 2] - p1[:, 0])
        # Compute cross product of vectors (p2[1] - p2[0]) x (p2[2] - p2[0])
        cross2 = torch.cross(p2[:, 1] - p2[:, 0], p2[:, 2] - p2[:, 0])
        
        # Compute dot product of cross products
        dot_product = torch.sum(cross1 * cross2, dim=-1)
        
        # Valid if dot product is positive
        return dot_product > 0
    
    # Apply the oriented constraint check for each batch
    validity_mask = oriented_constraint(points1_h, points2_h)
    
    # Return the validity mask as a tensor of shape (B, 1)
    return validity_mask.unsqueeze(-1)

