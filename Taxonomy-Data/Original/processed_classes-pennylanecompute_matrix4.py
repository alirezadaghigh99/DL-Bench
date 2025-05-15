    def compute_matrix(A):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.THermitian.matrix`

        Args:
            A (tensor_like): Hermitian matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[6+0j, 1-2j, 0],[1+2j, -1, 0], [0, 0, 1]])
        >>> qml.THermitian.compute_matrix(A)
        [[ 6.+0.j  1.-2.j  0.+0.j]
         [ 1.+2.j -1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j]]
        """
        return Hermitian.compute_matrix(A)