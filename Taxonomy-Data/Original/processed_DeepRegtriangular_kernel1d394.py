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

    padding = kernel_size // 2
    kernel = tf.constant(
        [0] * math.ceil(padding / 2)
        + [1] * (kernel_size - padding)
        + [0] * math.floor(padding / 2),
        dtype=tf.float32,
    )

    # (padding*2, )
    filters = tf.ones(shape=(kernel_size - padding, 1, 1), dtype=tf.float32)

    # (kernel_size, 1, 1)
    kernel = tf.nn.conv1d(
        kernel[None, :, None], filters=filters, stride=[1, 1, 1], padding="SAME"
    )

    return kernel[0, :, 0]