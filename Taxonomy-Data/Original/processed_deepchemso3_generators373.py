def so3_generators(k: int) -> torch.Tensor:
    """Construct the generators of the SO(3) Lie algebra for a given quantum angular momentum.

    The function generates the generators of the special orthogonal group SO(3), which represents the group
    of rotations in three-dimensional space. Its Lie algebra, which consists of the generators of
    infinitesimal rotations, is often used in physics to describe angular momentum operators.
    The generators of the Lie algebra can be related to the SU(2) group, and this function uses
    a transformation to convert the SU(2) generators to the SO(3) basis.

    The primary significance of the SO(3) group lies in its representation of three-dimensional
    rotations. Each matrix in SO(3) corresponds to a unique rotation, capturing the intricate
    ways in which objects can be oriented in 3D space. This concept finds application in
    numerous fields, ranging from physics to engineering.

    Parameters
    ----------
     k : int
        The representation index, which determines the order of the representation.

    Returns
    -------
    torch.Tensor
        A stack of three SO(3) generators, corresponding to J_x, J_z, and J_y.

    Notes
    -----
    The special orthogonal group $SO_n(q)$ is the subgroup of the elements of general orthogonal
    group $GO_n(q)$ with determinant 1. $SO_3$ (often written $SO(3)$) is the rotation group
    for three-dimensional space.

    These matrices are orthogonal, which means their rows and columns form mutually perpendicular
    unit vectors. This preservation of angles and lengths makes orthogonal matrices fundamental
    in various mathematical and practical applications.

    The "special" part of $SO(3)$ refers to the determinant of these matrices being $+1$. The
    determinant is a scalar value that indicates how much a matrix scales volumes.
    A determinant of $+1$ ensures that the matrix represents a rotation in three-dimensional
    space without involving any reflection or scaling operations that would reverse the orientation of space.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Special_orthogonal_group
    .. [2] https://en.wikipedia.org/wiki/3D_rotation_group#Connection_between_SO(3)_and_SU(2)
    .. [3] https://www.pas.rochester.edu/assets/pdf/undergraduate/su-2s_double_covering_of_so-3.pdf

    Examples
    --------
    >>> so3_generators(1)
    tensor([[[ 0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000, -1.0000],
             [ 0.0000,  1.0000,  0.0000]],
    <BLANKLINE>
            [[ 0.0000,  0.0000,  1.0000],
             [ 0.0000,  0.0000,  0.0000],
             [-1.0000,  0.0000,  0.0000]],
    <BLANKLINE>
            [[ 0.0000, -1.0000,  0.0000],
             [ 1.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000]]])
    """
    # Get the SU(2) generators for the given quantum angular momentum (spin) value.
    X = su2_generators(k)

    # Get the transformation matrix to change the basis from real to complex spherical harmonics.
    Q = change_basis_real_to_complex(k)

    # Convert the SU(2) generators to the SO(3) basis using the transformation matrix Q.
    # X represents the SU(2) generators, and Q is the transformation matrix from real to complex spherical harmonics.
    # The resulting X matrix will be the SO(3) generators in the complex basis.
    X = torch.conj(Q.T) @ X @ Q

    # Return the real part of the SO(3) generators to ensure they are purely real.
    return torch.real(X)