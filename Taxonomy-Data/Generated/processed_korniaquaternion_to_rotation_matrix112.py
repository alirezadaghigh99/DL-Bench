import torch

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    
    # Unpack the quaternion components
    w, x, y, z = quaternion.unbind(dim=-1)
    
    # Compute the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Construct the rotation matrix
    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(*quaternion.shape[:-1], 3, 3)
    
    return rotation_matrix

