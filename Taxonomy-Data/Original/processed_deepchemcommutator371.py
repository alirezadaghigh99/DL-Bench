def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the commutator of two matrices.

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

    Examples
    --------
    >>> A = torch.tensor([[1, 2], [3, 4]])
    >>> B = torch.tensor([[5, 6], [7, 8]])
    >>> commutator(A, B)
    tensor([[ -4, -12],
            [ 12,   4]])
    """
    return torch.matmul(A, B) - torch.matmul(B, A)