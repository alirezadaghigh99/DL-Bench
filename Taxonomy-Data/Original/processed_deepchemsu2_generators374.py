def su2_generators(k: int) -> torch.Tensor:
    """Generate the generators of the special unitary group SU(2) in a given representation.

    The function computes the generators of the SU(2) group for a specific representation
    determined by the value of 'k'. These generators are commonly used in the study of
    quantum mechanics, angular momentum, and related areas of physics and mathematics.
    The generators are represented as matrices.

    The SU(2) group is a fundamental concept in quantum mechanics and symmetry theory.
    The generators of the group, denoted as J_x, J_y, and J_z, represent the three
    components of angular momentum operators. These generators play a key role in
    describing the transformation properties of physical systems under rotations.

    The returned tensor contains three matrices corresponding to the x, y, and z generators,
    usually denoted as J_x, J_y, and J_z. These matrices form a basis for the Lie algebra
    of the SU(2) group.

    In linear algebra, specifically within the context of quantum mechanics, lowering and
    raising operators are fundamental concepts that play a crucial role in altering the
    eigenvalues of certain operators while acting on quantum states. These operators are
    often referred to collectively as "ladder operators."

    A lowering operator is an operator that, when applied to a quantum state, reduces the
    eigenvalue associated with a particular observable. In the context of SU(2), the lowering
    operator corresponds to J_-.

    Conversely, a raising operator is an operator that increases the eigenvalue of an
    observable when applied to a quantum state. In the context of SU(2), the raising operator
    corresponds to J_+.

    The z-generator matrix represents the component of angular momentum along the z-axis,
    often denoted as J_z. It commutes with both J_x and J_y and is responsible for quantizing
    the angular momentum.

    Note that the dimensions of the returned tensor will be (3, 2j+1, 2j+1), where each matrix
    has a size of (2j+1) x (2j+1).
    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.

    Returns
    -------
    torch.Tensor
        A stack of three SU(2) generators, corresponding to J_x, J_z, and J_y.

    Notes
    -----
    A generating set of a group is a subset $S$ of the group $G$ such that every element
    of $G$ can be expressed as a combination (under the group operation) of finitely many
    elements of the subset $S$ and their inverses.

    The special unitary group $SU_n(q)$ is the set of $n*n$ unitary matrices with determinant
    +1. $SU(2)$ is homeomorphic with the orthogonal group $O_3^+(2)$. It is also called the
    unitary unimodular group and is a Lie group.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Ladder_operator
    .. [2] https://en.wikipedia.org/wiki/Special_unitary_group#The_group_SU(2)
    .. [3] https://en.wikipedia.org/wiki/Generating_set_of_a_group
    .. [4] https://mathworld.wolfram.com/SpecialUnitaryGroup

    Examples
    --------
    >>> su2_generators(1)
    tensor([[[ 0.0000+0.0000j,  0.7071+0.0000j,  0.0000+0.0000j],
             [-0.7071+0.0000j,  0.0000+0.0000j,  0.7071+0.0000j],
             [ 0.0000+0.0000j, -0.7071+0.0000j,  0.0000+0.0000j]],
    <BLANKLINE>
            [[-0.0000-1.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+1.0000j]],
    <BLANKLINE>
            [[ 0.0000-0.0000j,  0.0000+0.7071j,  0.0000-0.0000j],
             [ 0.0000+0.7071j,  0.0000-0.0000j,  0.0000+0.7071j],
             [ 0.0000-0.0000j,  0.0000+0.7071j,  0.0000-0.0000j]]])
    """
    # Generate the raising operator matrix
    m = torch.arange(-k, k)
    raising = torch.diag(-torch.sqrt(k * (k + 1) - m * (m + 1)), diagonal=-1)

    # Generate the lowering operator matrix
    m = torch.arange(-k + 1, k + 1)
    lowering = torch.diag(torch.sqrt(k * (k + 1) - m * (m - 1)), diagonal=1)

    # Generate the z-generator matrix
    m = torch.arange(-k, k + 1)
    z_generator = torch.diag(1j * m)

    # Combine the matrices to form the x, z, and y generators
    x_generator = 0.5 * (raising + lowering)  # x (usually)
    y_generator = -0.5j * (raising - lowering)  # -y (usually)

    # Stack the generators along the first dimension to create a tensor
    generators = torch.stack([x_generator, z_generator, y_generator], dim=0)

    return generators