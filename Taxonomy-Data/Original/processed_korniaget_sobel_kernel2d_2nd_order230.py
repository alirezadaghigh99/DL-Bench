def get_sobel_kernel2d_2nd_order(*, device: Optional[Device] = None, dtype: Optional[Dtype] = None) -> Tensor:
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return stack([gxx, gxy, gyy])