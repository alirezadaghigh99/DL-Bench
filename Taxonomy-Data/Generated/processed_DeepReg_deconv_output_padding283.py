def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    if padding not in {"same", "valid", "full"}:
        raise ValueError(f"Unknown padding type: {padding}")

    # Calculate the expected output length without output padding
    if padding == "same":
        expected_output_length = (input_shape - 1) * stride + 1
    elif padding == "valid":
        expected_output_length = (input_shape - 1) * stride + kernel_size
    elif padding == "full":
        expected_output_length = (input_shape - 1) * stride + 2 * (kernel_size - 1) + 1

    # Calculate the output padding needed to reach the desired output shape
    output_padding = output_shape - expected_output_length

    if output_padding < 0:
        raise ValueError("Calculated output shape is larger than the desired output shape. Check the parameters.")

    return output_padding

