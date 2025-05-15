import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Create a 3d separable filter.

    :param tensor: shape = (batch, dim1, dim2, dim3, 1)
    :param kernel: shape = (dim4,)
    :return: shape = (batch, dim1, dim2, dim3, 1)
    """
    # Reshape the kernel to be used for 1D convolution
    kernel = tf.reshape(kernel, [-1, 1, 1, 1, 1])

    # Convolve along the depth (dim1) axis
    conv_depth = tf.nn.conv3d(tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

    # Convolve along the height (dim2) axis
    conv_height = tf.nn.conv3d(conv_depth, tf.transpose(kernel, [1, 0, 2, 3, 4]), strides=[1, 1, 1, 1, 1], padding='SAME')

    # Convolve along the width (dim3) axis
    conv_width = tf.nn.conv3d(conv_height, tf.transpose(kernel, [1, 2, 0, 3, 4]), strides=[1, 1, 1, 1, 1], padding='SAME')

    return conv_width