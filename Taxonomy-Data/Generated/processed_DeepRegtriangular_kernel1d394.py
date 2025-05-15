import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D triangular kernel for LocalNormalizedCrossCorrelation.

    Assume kernel_size is odd, it will be a smoothed from
    a kernel which center part is zero.
    Then length of the ones will be around half kernel_size.
    The weight scale of the kernel does not matter as LNCC will normalize it.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """

    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    # Calculate the half size of the kernel
    half_size = kernel_size // 2

    # Create a triangular kernel
    kernel_weights = tf.range(half_size + 1, dtype=tf.float32)
    kernel_weights = tf.concat([kernel_weights, tf.reverse(kernel_weights[:-1], axis=[0])], axis=0)

    return kernel_weights

