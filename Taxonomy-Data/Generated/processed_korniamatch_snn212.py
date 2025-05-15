import torch

def match_snn(desc1, desc2, th=0.8, dm=None):
    # Validate input shapes
    if desc1.ndim != 2 or desc2.ndim != 2:
        raise ValueError("desc1 and desc2 must be 2D tensors.")
    
    if desc2.size(0) < 2:
        # If desc2 has fewer than two descriptors, return empty results
        return torch.empty(0, 1), torch.empty(0, 2, dtype=torch.long)
    
    # Calculate the distance matrix if not provided
    if dm is None:
        # Compute pairwise L2 distances
        dm = torch.cdist(desc1, desc2, p=2)
    
    # Find the nearest and second nearest neighbors
    sorted_distances, sorted_indices = torch.sort(dm, dim=1)
    
    # Nearest and second nearest distances and indices
    nearest_distances = sorted_distances[:, 0]
    second_nearest_distances = sorted_distances[:, 1]
    nearest_indices = sorted_indices[:, 0]
    
    # Apply the ratio test
    ratio = nearest_distances / second_nearest_distances
    valid_matches = ratio <= th
    
    # Filter valid matches
    matched_distances = nearest_distances[valid_matches].unsqueeze(1)
    matched_indices = torch.stack((torch.arange(desc1.size(0))[valid_matches], nearest_indices[valid_matches]), dim=1)
    
    return matched_distances, matched_indices

