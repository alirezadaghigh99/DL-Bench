import tensorflow as tf
from typing import Sequence, Tuple

def reshape_Z_dZ(
    zs: Sequence[tf.Tensor], dzs: Sequence[tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    :param zs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :param dzs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :returns: points Z, Tensor with shape [N1*N2*...*Nd, D],
        and weights dZ, Tensor with shape [N1*N2*...*Nd, 1]
    """
    # Ensure that zs and dzs have the same length
    assert len(zs) == len(dzs), "zs and dzs must have the same length"
    
    # Get the number of dimensions
    D = len(zs)
    
    # Create a meshgrid from the zs
    meshgrid = tf.meshgrid(*zs, indexing='ij')
    
    # Flatten each grid and stack them to form Z
    Z = tf.stack([tf.reshape(grid, [-1]) for grid in meshgrid], axis=-1)
    
    # Compute the product of all dzs to form dZ
    dZ = tf.reduce_prod(tf.stack(tf.meshgrid(*dzs, indexing='ij'), axis=-1), axis=-1)
    dZ = tf.reshape(dZ, [-1, 1])
    
    return Z, dZ