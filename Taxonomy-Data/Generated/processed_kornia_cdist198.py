import torch

def _cdist(d1, d2):
    # Check if tensors are compatible with MPS
    def is_mps_compatible(tensor):
        return tensor.is_floating_point() and tensor.device.type == 'mps'

    # If both tensors are not MPS compatible, use torch.cdist directly
    if not (is_mps_compatible(d1) and is_mps_compatible(d2)):
        return torch.cdist(d1, d2)

    # Manual computation of pairwise Euclidean distance
    # Compute squared norms of each row in d1 and d2
    d1_squared = torch.sum(d1 ** 2, dim=1, keepdim=True)
    d2_squared = torch.sum(d2 ** 2, dim=1, keepdim=True)

    # Compute the matrix multiplication part
    # d1 @ d2.T gives the dot product between each pair of rows
    dot_product = torch.mm(d1, d2.t())

    # Compute the pairwise squared distances
    # Using broadcasting to add the squared norms and subtract the dot product
    pairwise_squared_dist = d1_squared + d2_squared.t() - 2 * dot_product

    # Clamp the result to ensure no negative distances due to floating point errors
    pairwise_squared_dist = torch.clamp(pairwise_squared_dist, min=0.0)

    # Return the square root of the squared distances to get the Euclidean distances
    return torch.sqrt(pairwise_squared_dist)

