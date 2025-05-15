import torch
from scipy.optimize import linear_sum_assignment

def maximum_weight_matching(logits):
    """
    Find the maximum-weight matching for the given logits tensor.

    Args:
        logits (torch.Tensor): A 2D tensor of shape (n, m) representing the weights of the bipartite graph.

    Returns:
        torch.Tensor: A 1D tensor of shape (n,) where each element is the index of the matched column for each row.
    """
    # Convert logits to numpy array for processing with scipy
    logits_np = logits.detach().cpu().numpy()

    # Since linear_sum_assignment finds the minimum cost, we need to negate the weights for maximum weight matching
    cost_matrix = -logits_np

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert the result back to a torch tensor
    assignment = torch.tensor(col_ind, dtype=torch.int64)

    return assignment

