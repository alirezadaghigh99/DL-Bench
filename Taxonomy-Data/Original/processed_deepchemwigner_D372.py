def wigner_D(k: int, alpha: torch.Tensor, beta: torch.Tensor,
             gamma: torch.Tensor) -> torch.Tensor:
    """Wigner D matrix representation of the SO(3) rotation group.

    The function computes the Wigner D matrix representation of the SO(3) rotation group
    for a given representation index 'k' and rotation angles 'alpha', 'beta', and 'gamma'.
    The resulting matrix satisfies properties of the SO(3) group representation.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    alpha : torch.Tensor
        Rotation angles (in radians) around the Y axis, applied third.
    beta : torch.Tensor
        Rotation angles (in radians) around the X axis, applied second.
    gamma : torch.Tensor)
        Rotation angles (in radians) around the Y axis, applied first.

    Returns
    -------
    torch.Tensor
        The Wigner D matrix of shape (#angles, 2k+1, 2k+1).

    Notes
    -----
    The Wigner D-matrix is a unitary matrix in an irreducible representation
    of the groups SU(2) and SO(3).

    The Wigner D-matrix is used in quantum mechanics to describe the action
    of rotations on states of particles with angular momentum. It is a key
    concept in the representation theory of the rotation group SO(3), and
    it plays a crucial role in various physical contexts.

    Examples
    --------
    >>> k = 1
    >>> alpha = torch.tensor([0.1, 0.2])
    >>> beta = torch.tensor([0.3, 0.4])
    >>> gamma = torch.tensor([0.5, 0.6])
    >>> wigner_D_matrix = wigner_D(k, alpha, beta, gamma)
    >>> wigner_D_matrix
    tensor([[[ 0.8275,  0.1417,  0.5433],
             [ 0.0295,  0.9553, -0.2940],
             [-0.5607,  0.2593,  0.7863]],
    <BLANKLINE>
            [[ 0.7056,  0.2199,  0.6737],
             [ 0.0774,  0.9211, -0.3816],
             [-0.7044,  0.3214,  0.6329]]])
    """
    # Ensure that alpha, beta, and gamma have the same shape for broadcasting.
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)

    # Ensure the angles are within the range [0, 2*pi) using modulo.
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    # Get the SO(3) generators for the given quantum angular momentum (spin) value 'k'.
    X = so3_generators(k)

    # Calculate the Wigner D matrix using the matrix exponential of the generators
    # and the rotation angles alpha, beta, and gamma in the appropriate order.
    D_matrix = torch.matrix_exp(gamma * (X[1].unsqueeze(0))) @ torch.matrix_exp(
        beta * (X[0].unsqueeze(0))) @ torch.matrix_exp(alpha *
                                                       (X[1].unsqueeze(0)))
    return D_matrix