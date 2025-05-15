def gaussian_filter_3d(kernel_sigma: Union[Tuple, List]) -> tf.Tensor:
    """
    Define a gaussian filter in 3d for smoothing.

    The filter size is defined 3*kernel_sigma


    :param kernel_sigma: the deviation at each direction (list)
        or use an isotropic deviation (int)
    :return: kernel: tf.Tensor specify a gaussian kernel of shape:
        [3*k for k in kernel_sigma]
    """
    if isinstance(kernel_sigma, (int, float)):
        kernel_sigma = (kernel_sigma, kernel_sigma, kernel_sigma)

    kernel_size = [
        int(np.ceil(ks * 3) + np.mod(np.ceil(ks * 3) + 1, 2)) for ks in kernel_sigma
    ]

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    coord = [np.arange(ks) for ks in kernel_size]

    xx, yy, zz = np.meshgrid(coord[0], coord[1], coord[2], indexing="ij")
    xyz_grid = np.concatenate(
        (xx[np.newaxis], yy[np.newaxis], zz[np.newaxis]), axis=0
    )  # 2, y, x

    mean = np.asarray([(ks - 1) / 2.0 for ks in kernel_size])
    mean = mean.reshape(-1, 1, 1, 1)
    variance = np.asarray([ks ** 2.0 for ks in kernel_sigma])
    variance = variance.reshape(-1, 1, 1, 1)

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    # 2.506628274631 = sqrt(2 * pi)

    norm_kernel = 1.0 / (np.sqrt(2 * np.pi) ** 3 + np.prod(kernel_sigma))
    kernel = norm_kernel * np.exp(
        -np.sum((xyz_grid - mean) ** 2.0 / (2 * variance), axis=0)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / np.sum(kernel)

    # Reshape
    kernel = kernel.reshape(kernel_size[0], kernel_size[1], kernel_size[2])

    # Total kernel
    total_kernel = np.zeros(tuple(kernel_size) + (3, 3))
    total_kernel[..., 0, 0] = kernel
    total_kernel[..., 1, 1] = kernel
    total_kernel[..., 2, 2] = kernel

    return tf.convert_to_tensor(total_kernel, dtype=tf.float32)