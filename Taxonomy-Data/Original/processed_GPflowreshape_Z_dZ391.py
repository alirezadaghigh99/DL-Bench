def reshape_Z_dZ(
    zs: Sequence[TensorType], dzs: Sequence[TensorType]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    :param zs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :param dzs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :returns: points Z, Tensor with shape [N1*N2*...*Nd, D],
        and weights dZ, Tensor with shape [N1*N2*...*Nd, 1]
    """
    Z = list_to_flat_grid(zs)
    dZ = tf.reduce_prod(list_to_flat_grid(dzs), axis=-1, keepdims=True)
    return Z, dZ