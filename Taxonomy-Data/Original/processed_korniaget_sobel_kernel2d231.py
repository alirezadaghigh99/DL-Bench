def get_sobel_kernel2d(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])