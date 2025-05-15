import torch

def laf_from_center_scale_ori(xy, scale=None, ori=None):
    # Check the shape of the input tensor xy
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError("Input tensor xy must have shape (B, N, 2)")

    # Initialize device and data type
    device = xy.device
    dtype = xy.dtype

    # Calculate batch size B and number of keypoints N
    B, N, _ = xy.shape

    # If scale is not provided, set it to ones tensor of the appropriate shape
    if scale is None:
        scale = torch.ones((B, N, 1, 1), device=device, dtype=dtype)
    else:
        if scale.shape != (B, N, 1, 1):
            raise ValueError("Scale tensor must have shape (B, N, 1, 1)")

    # If orientation is not provided, set it to zeros tensor of the appropriate shape
    if ori is None:
        ori = torch.zeros((B, N, 1), device=device, dtype=dtype)
    else:
        if ori.shape != (B, N, 1):
            raise ValueError("Orientation tensor must have shape (B, N, 1)")

    # Calculate the rotation matrix based on the orientation
    cos_ori = torch.cos(ori)
    sin_ori = torch.sin(ori)
    rotation_matrix = torch.cat([
        cos_ori, -sin_ori,
        sin_ori, cos_ori
    ], dim=-1).view(B, N, 2, 2)

    # Concatenate the rotation matrix with the keypoint centers
    laf = torch.cat([rotation_matrix, xy.unsqueeze(-1)], dim=-1)

    # Scale the LAF based on the provided scale
    laf = laf * scale

    return laf

