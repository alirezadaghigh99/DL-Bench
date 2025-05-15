import tensorflow as tf
import numpy as np

def gaussian_filter_3d(kernel_sigma):
    # Ensure kernel_sigma is a list of three elements
    if isinstance(kernel_sigma, int):
        kernel_sigma = [kernel_sigma] * 3
    elif isinstance(kernel_sigma, list) and len(kernel_sigma) != 3:
        raise ValueError("kernel_sigma must be an integer or a list of three elements.")

    # Calculate the size of the kernel in each dimension
    kernel_size = [3 * int(sigma) for sigma in kernel_sigma]

    # Create a 3D grid of (x, y, z) coordinates
    x = np.arange(-kernel_size[0] // 2 + 1, kernel_size[0] // 2 + 1)
    y = np.arange(-kernel_size[1] // 2 + 1, kernel_size[1] // 2 + 1)
    z = np.arange(-kernel_size[2] // 2 + 1, kernel_size[2] // 2 + 1)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate the Gaussian function
    gaussian_kernel = np.exp(
        -((x**2) / (2 * kernel_sigma[0]**2) +
          (y**2) / (2 * kernel_sigma[1]**2) +
          (z**2) / (2 * kernel_sigma[2]**2))
    )

    # Normalize the kernel so that the sum is 1
    gaussian_kernel /= np.sum(gaussian_kernel)

    # Convert the kernel to a TensorFlow tensor
    gaussian_kernel_tensor = tf.convert_to_tensor(gaussian_kernel, dtype=tf.float32)

    return gaussian_kernel_tensor

