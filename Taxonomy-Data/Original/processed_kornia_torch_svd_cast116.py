def _torch_svd_cast(input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Helper function to make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    # if not isinstance(input, torch.Tensor):
    #    raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    out1, out2, out3H = torch.linalg.svd(input.to(dtype))
    if torch_version_ge(1, 11):
        out3 = out3H.mH
    else:
        out3 = out3H.transpose(-1, -2)
    return (out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype))