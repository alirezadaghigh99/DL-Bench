import tensorflow as tf
import numpy as np

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    """
    Approximates a 1D Cauchy kernel.

    :param sigma: int, defining the scale parameter of the Cauchy distribution.
    :return: A 1D tensor representing the Cauchy kernel.
    """
    assert sigma > 0, "Sigma must be greater than 0"

    # Define the size of the kernel. A common choice is to use 6 times the scale parameter.
    size = int(6 * sigma)
    if size % 2 == 0:
        size += 1  # Ensure the size is odd for symmetry

    # Create a range of values centered around zero
    x = np.arange(-size // 2, size // 2 + 1, 1)

    # Calculate the Cauchy kernel
    kernel = 1 / (np.pi * sigma * (1 + (x / sigma) ** 2))

    # Normalize the kernel to ensure the sum is 1
    kernel /= np.sum(kernel)

    # Convert the kernel to a TensorFlow tensor
    kernel_tensor = tf.convert_to_tensor(kernel, dtype=tf.float32)

    return kernel_tensor

