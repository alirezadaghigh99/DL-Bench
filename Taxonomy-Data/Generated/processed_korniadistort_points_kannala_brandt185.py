import torch

def distort_points_kannala_brandt(projected_points_in_camera_z1_plane, params):
    # Unpack the parameters
    fx, fy, cx, cy, k1, k2, k3, k4 = params[..., 0], params[..., 1], params[..., 2], params[..., 3], params[..., 4], params[..., 5], params[..., 6], params[..., 7]
    
    # Unpack the points
    x, y = projected_points_in_camera_z1_plane[..., 0], projected_points_in_camera_z1_plane[..., 1]
    
    # Compute the radial distance from the center
    r = torch.sqrt(x**2 + y**2)
    
    # Compute the distortion factor using the polynomial
    theta = torch.atan(r)
    theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    
    # Avoid division by zero
    r = torch.where(r == 0, torch.tensor(1e-8, dtype=r.dtype, device=r.device), r)
    
    # Scale the distorted radius
    scale = theta_d / r
    
    # Apply the distortion
    x_distorted = scale * x
    y_distorted = scale * y
    
    # Convert to pixel coordinates
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    # Stack the results
    distorted_points = torch.stack((u, v), dim=-1)
    
    return distorted_points

