import torch

def symeig3x3(matrix: torch.Tensor, eigenvectors: bool = False):
    """
    Computes the eigenvalues and optionally the eigenvectors of symmetric 3x3 matrices.

    Parameters:
    - matrix (torch.Tensor): A tensor of shape (..., 3, 3) representing symmetric matrices.
    - eigenvectors (bool): If True, compute both eigenvalues and eigenvectors. If False, compute only eigenvalues.

    Returns:
    - If eigenvectors is False, returns a tensor of eigenvalues with shape (..., 3).
    - If eigenvectors is True, returns a tuple (eigenvalues, eigenvectors) where:
      - eigenvalues is a tensor of shape (..., 3).
      - eigenvectors is a tensor of shape (..., 3, 3).
    """
    # Ensure the input is a symmetric matrix
    if not torch.allclose(matrix, matrix.transpose(-2, -1)):
        raise ValueError("Input matrices must be symmetric.")

    # Use torch.linalg.eigh to compute eigenvalues and eigenvectors
    if eigenvectors:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return eigenvalues, eigenvectors
    else:
        eigenvalues = torch.linalg.eigvalsh(matrix)
        return eigenvalues

