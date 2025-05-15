import torch

def _no_match(B1):
    # Returns placeholder outputs when there are no matches
    return torch.full((B1, 1), float('inf')), torch.full((B1, 2), -1, dtype=torch.long)

def match_nn(desc1, desc2, dm=None):
    # Check if either descriptor is empty
    if desc1.size(0) == 0 or desc2.size(0) == 0:
        return _no_match(desc1.size(0))
    
    # Calculate the distance matrix if not provided
    if dm is None:
        dm = torch.cdist(desc1, desc2)
    
    # Find the nearest neighbors
    min_distances, min_indices = torch.min(dm, dim=1)
    
    # Construct the indices tensor
    indices = torch.stack((torch.arange(desc1.size(0)), min_indices), dim=1)
    
    # Return the distances and indices
    return min_distances.unsqueeze(1), indices

