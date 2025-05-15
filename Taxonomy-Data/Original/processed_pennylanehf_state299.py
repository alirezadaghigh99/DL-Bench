def hf_state(electrons, orbitals, basis="occupation_number"):
    r"""Generate the Hartree-Fock statevector with respect to a chosen basis.

    The many-particle wave function in the Hartree-Fock (HF) approximation is a `Slater determinant
    <https://en.wikipedia.org/wiki/Slater_determinant>`_. In Fock space, a Slater determinant
    for :math:`N` electrons is represented by the occupation-number vector:

    .. math::

        \vert {\bf n} \rangle = \vert n_1, n_2, \dots, n_\mathrm{orbs} \rangle,
        n_i = \left\lbrace \begin{array}{ll} 1 & i \leq N \\ 0 & i > N \end{array} \right.,

    where :math:`n_i` indicates the occupation of the :math:`i`-th orbital.

    The Hartree-Fock state can also be generated in the parity basis, where each qubit stores the parity of
    the spin orbital, and in the Bravyi-Kitaev basis, where a qubit :math:`j` stores the occupation state of orbital
    :math:`j` if :math:`j` is even and stores partial sum of the occupation state of a set of orbitals of indices
    less than :math:`j` if :math:`j` is odd [`Tranter et al. Int. J. Quantum Chem. 115, 1431 (2015)
    <https://doi.org/10.1002/qua.24969>`_].

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,
            this is the number of active spin-orbitals.
        basis (string): Basis in which the HF state is represented. Options are ``occupation_number``, ``parity`` and ``bravyi_kitaev``.

    Returns:
        array: NumPy array containing the vector :math:`\vert {\bf n} \rangle`

    **Example**

    >>> state = hf_state(2, 6)
    >>> print(state)
    [1 1 0 0 0 0]

    >>> state = hf_state(2, 6, basis="parity")
    >>> print(state)
    [1 0 0 0 0 0]

    >>> state = hf_state(2, 6, basis="bravyi_kitaev")
    >>> print(state)
    [1 0 0 0 0 0]

    """

    if electrons <= 0:
        raise ValueError(
            f"The number of active electrons has to be larger than zero; "
            f"got 'electrons' = {electrons}"
        )

    if electrons > orbitals:
        raise ValueError(
            f"The number of active orbitals cannot be smaller than the number of active electrons;"
            f" got 'orbitals'={orbitals} < 'electrons'={electrons}"
        )

    state = np.where(np.arange(orbitals) < electrons, 1, 0)

    if basis == "parity":
        pi_matrix = np.tril(np.ones((orbitals, orbitals)))
        return (np.matmul(pi_matrix, state) % 2).astype(int)

    if basis == "bravyi_kitaev":
        beta_matrix = _beta_matrix(orbitals)
        return (np.matmul(beta_matrix, state) % 2).astype(int)

    return state