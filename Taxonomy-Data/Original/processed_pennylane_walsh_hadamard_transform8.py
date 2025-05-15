def _walsh_hadamard_transform(D: TensorLike, n: Optional[int] = None):
    r"""Compute the Walsh–Hadamard Transform of a one-dimensional array.

    Args:
        D (tensor_like): The array or tensor to be transformed. Must have a length that
            is a power of two.

    Returns:
        tensor_like: The transformed tensor with the same shape as the input ``D``.

    Due to the execution of the transform as a sequence of tensor multiplications
    with shapes ``(2, 2), (2, 2,... 2)->(2, 2,... 2)``, the theoretical scaling of this
    method is the same as the one for the
    `Fast Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Fast_Walsh-Hadamard_transform>`__:
    On ``n`` qubits, there are ``n`` calls to ``tensordot``, each multiplying a
    ``(2, 2)`` matrix to a ``(2,)*n`` vector, with a single axis being contracted. This means
    that there are ``n`` operations with a FLOP count of ``4 * 2**(n-1)``, where ``4`` is the cost
    of a single ``(2, 2) @ (2,)`` contraction and ``2**(n-1)`` is the number of copies due to the
    non-contracted ``n-1`` axes.
    Due to the large internal speedups of compiled matrix multiplication and compatibility
    with autodifferentiation frameworks, the approach taken here is favourable over a manual
    realization of the FWHT unless memory limitations restrict the creation of intermediate
    arrays.
    """
    orig_shape = qml.math.shape(D)
    n = n or int(qml.math.log2(orig_shape[-1]))
    # Reshape the array so that we may apply the Hadamard transform to each axis individually
    if broadcasted := len(orig_shape) > 1:
        new_shape = (orig_shape[0],) + (2,) * n
    else:
        new_shape = (2,) * n
    D = qml.math.reshape(D, new_shape)
    # Apply Hadamard transform to each axis, shifted by one for broadcasting
    for i in range(broadcasted, n + broadcasted):
        D = qml.math.tensordot(_walsh_hadamard_matrix, D, axes=[[1], [i]])
    # The axes are in reverted order after all matrix multiplications, so we need to transpose;
    # If D was broadcasted, this moves the broadcasting axis to first position as well.
    # Finally, reshape to original shape
    return qml.math.reshape(qml.math.transpose(D), orig_shape)