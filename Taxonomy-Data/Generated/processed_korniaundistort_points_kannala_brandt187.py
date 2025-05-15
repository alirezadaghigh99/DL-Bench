import torch

def undistort_points_kannala_brandt(distorted_points_in_camera, params, max_iterations=10, tolerance=1e-6):
    """
    Undistorts points using the Kannala-Brandt model and Gauss-Newton optimization.

    Args:
        distorted_points_in_camera (Tensor): Distorted points with shape (..., 2).
        params (Tensor): Parameters of the Kannala-Brandt model with shape (..., 8).
        max_iterations (int): Maximum number of iterations for the optimization.
        tolerance (float): Tolerance for convergence.

    Returns:
        Tensor: Undistorted points with shape (..., 2).
    """
    # Extract parameters
    fx, fy, cx, cy, k1, k2, k3, k4 = torch.split(params, 1, dim=-1)

    # Initial guess for undistorted points
    undistorted_points = distorted_points_in_camera.clone()

    for _ in range(max_iterations):
        # Compute the current distorted points from the undistorted guess
        x = undistorted_points[..., 0:1]
        y = undistorted_points[..., 1:2]
        
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan(r)
        
        theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        
        scale = torch.where(r > 0, theta_d / r, torch.ones_like(r))
        
        x_distorted = fx * x * scale + cx
        y_distorted = fy * y * scale + cy
        
        # Compute the error
        error_x = x_distorted - distorted_points_in_camera[..., 0:1]
        error_y = y_distorted - distorted_points_in_camera[..., 1:2]
        
        # Check for convergence
        error = torch.cat([error_x, error_y], dim=-1)
        if torch.max(torch.abs(error)) < tolerance:
            break
        
        # Compute the Jacobian
        dtheta_d_dtheta = 1 + 3 * k1 * theta**2 + 5 * k2 * theta**4 + 7 * k3 * theta**6 + 9 * k4 * theta**8
        dscale_dr = (dtheta_d_dtheta * r - theta_d) / (r**2)
        
        dx_distorted_dx = fx * (scale + x**2 * dscale_dr / r)
        dx_distorted_dy = fx * (x * y * dscale_dr / r)
        dy_distorted_dx = fy * (x * y * dscale_dr / r)
        dy_distorted_dy = fy * (scale + y**2 * dscale_dr / r)
        
        # Construct the Jacobian matrix
        J = torch.stack([
            torch.cat([dx_distorted_dx, dx_distorted_dy], dim=-1),
            torch.cat([dy_distorted_dx, dy_distorted_dy], dim=-1)
        ], dim=-2)
        
        # Update the undistorted points using Gauss-Newton
        J_inv = torch.linalg.pinv(J)
        delta = torch.matmul(J_inv, error.unsqueeze(-1)).squeeze(-1)
        undistorted_points = undistorted_points - delta

    return undistorted_points