def _deconv_output_padding(
    input_shape: int, output_shape: int, kernel_size: int, stride: int, padding: str
) -> int:
    """
    Calculate output padding for Conv3DTranspose in 1D.

    - output_shape = (input_shape - 1)*stride + kernel_size - 2*pad + output_padding
    - output_padding = output_shape - ((input_shape - 1)*stride + kernel_size - 2*pad)

    Reference:

    - https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/python/keras/utils/conv_utils.py#L140

    :param input_shape: shape of Conv3DTranspose input tensor
    :param output_shape: shape of Conv3DTranspose output tensor
    :param kernel_size: kernel size of Conv3DTranspose layer
    :param stride: stride of Conv3DTranspose layer
    :param padding: padding of Conv3DTranspose layer
    :return: output_padding for Conv3DTranspose layer
    """
    if padding == "same":
        pad = kernel_size // 2
    elif padding == "valid":
        pad = 0
    elif padding == "full":
        pad = kernel_size - 1
    else:
        raise ValueError(f"Unknown padding {padding} in deconv_output_padding")
    return output_shape - ((input_shape - 1) * stride + kernel_size - 2 * pad)