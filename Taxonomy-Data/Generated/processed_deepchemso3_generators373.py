import torch

def so3_generators(k):
    """
    Generate the generators of the special orthogonal group SO(3).

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.

    Returns
    -------
    torch.Tensor
        A stack of three SO(3) generators, corresponding to J_x, J_y, and J_z.
    """
    if k < 1:
        raise ValueError("The representation index k must be a positive integer.")

    # Define the basic angular momentum matrices for the fundamental representation
    J_x = torch.zeros((2*k+1, 2*k+1), dtype=torch.float32)
    J_y = torch.zeros((2*k+1, 2*k+1), dtype=torch.float32)
    J_z = torch.zeros((2*k+1, 2*k+1), dtype=torch.float32)

    # Fill in the matrices
    for m in range(-k, k+1):
        if m < k:
            J_x[k+m, k+m+1] = 0.5 * torch.sqrt((k-m) * (k+m+1))
            J_x[k+m+1, k+m] = 0.5 * torch.sqrt((k-m) * (k+m+1))
            J_y[k+m, k+m+1] = -0.5j * torch.sqrt((k-m) * (k+m+1))
            J_y[k+m+1, k+m] = 0.5j * torch.sqrt((k-m) * (k+m+1))
        J_z[k+m, k+m] = m

    # Stack the generators into a single tensor
    generators = torch.stack((J_x, J_y, J_z))

    return generators

