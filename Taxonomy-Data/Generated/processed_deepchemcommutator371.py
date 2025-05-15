import torch

def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the commutator of two matrices.

    Parameters
    ----------
    A : torch.Tensor
        The first matrix.
    B : torch.Tensor
        The second matrix.

    Returns
    -------
    torch.Tensor
        The commutator of the two matrices.
    """
    # Ensure the input matrices are 2D tensors
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("Both A and B must be 2D matrices.")
    
    # Ensure the matrices have compatible dimensions for multiplication
    if A.size(1) != B.size(0) or B.size(1) != A.size(0):
        raise ValueError("Matrices A and B must have compatible dimensions for multiplication.")
    
    # Compute the commutator
    commutator_result = torch.mm(A, B) - torch.mm(B, A)
    
    return commutator_result