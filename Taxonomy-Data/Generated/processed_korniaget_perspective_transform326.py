import torch

def get_perspective_transform(points_src, points_dst):
    """
    Calculate a perspective transform from four pairs of the corresponding points using DLT.

    Args:
        points_src: coordinates of quadrangle vertices in the source image with shape (B, 4, 2).
        points_dst: coordinates of the corresponding quadrangle vertices in the destination image with shape (B, 4, 2).

    Returns:
        The perspective transformation with shape (B, 3, 3).
    """
    assert points_src.shape == points_dst.shape, "Source and destination points must have the same shape"
    assert points_src.shape[1:] == (4, 2), "Each set of points must have shape (4, 2)"

    batch_size = points_src.shape[0]
    A = torch.zeros((batch_size, 8, 8), dtype=points_src.dtype, device=points_src.device)
    b = torch.zeros((batch_size, 8, 1), dtype=points_src.dtype, device=points_src.device)

    for i in range(4):
        x_src, y_src = points_src[:, i, 0], points_src[:, i, 1]
        x_dst, y_dst = points_dst[:, i, 0], points_dst[:, i, 1]

        A[:, 2 * i, 0] = x_src
        A[:, 2 * i, 1] = y_src
        A[:, 2 * i, 2] = 1
        A[:, 2 * i, 6] = -x_src * x_dst
        A[:, 2 * i, 7] = -y_src * x_dst
        b[:, 2 * i, 0] = x_dst

        A[:, 2 * i + 1, 3] = x_src
        A[:, 2 * i + 1, 4] = y_src
        A[:, 2 * i + 1, 5] = 1
        A[:, 2 * i + 1, 6] = -x_src * y_dst
        A[:, 2 * i + 1, 7] = -y_src * y_dst
        b[:, 2 * i + 1, 0] = y_dst

    # Solve the system of equations A * h = b
    h = torch.linalg.solve(A, b)

    # Reshape h to (B, 3, 3) and add the last row [0, 0, 1]
    h = torch.cat([h, torch.ones((batch_size, 1, 1), dtype=points_src.dtype, device=points_src.device)], dim=1)
    h = h.view(batch_size, 3, 3)

    return h

