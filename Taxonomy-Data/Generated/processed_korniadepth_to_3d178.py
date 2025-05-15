import torch

def depth_to_3d(depth_tensor, camera_matrix, normalize_points=False):
    # Check input types
    if not isinstance(depth_tensor, torch.Tensor):
        raise TypeError("depth_tensor must be a torch.Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a torch.Tensor")
    
    # Check input shapes
    if depth_tensor.ndim != 4 or depth_tensor.shape[1] != 1:
        raise ValueError("depth_tensor must have shape (B, 1, H, W)")
    if camera_matrix.ndim != 3 or camera_matrix.shape[1:] != (3, 3):
        raise ValueError("camera_matrix must have shape (B, 3, 3)")
    
    B, _, H, W = depth_tensor.shape
    
    # Create a meshgrid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=depth_tensor.device), 
                          torch.arange(W, device=depth_tensor.device), 
                          indexing='ij')
    y = y.expand(B, -1, -1)
    x = x.expand(B, -1, -1)
    
    # Flatten the pixel coordinates
    x_flat = x.reshape(B, -1)
    y_flat = y.reshape(B, -1)
    
    # Get the depth values
    depth_flat = depth_tensor.reshape(B, -1)
    
    # Get the camera intrinsics
    fx = camera_matrix[:, 0, 0]
    fy = camera_matrix[:, 1, 1]
    cx = camera_matrix[:, 0, 2]
    cy = camera_matrix[:, 1, 2]
    
    # Compute the 3D points
    X = (x_flat - cx.unsqueeze(1)) * depth_flat / fx.unsqueeze(1)
    Y = (y_flat - cy.unsqueeze(1)) * depth_flat / fy.unsqueeze(1)
    Z = depth_flat
    
    # Stack the 3D points
    points_3d = torch.stack((X, Y, Z), dim=1)
    
    # Reshape to (B, 3, H, W)
    points_3d = points_3d.reshape(B, 3, H, W)
    
    if normalize_points:
        # Normalize the 3D points
        norm = torch.norm(points_3d, dim=1, keepdim=True)
        points_3d = points_3d / (norm + 1e-8)  # Add epsilon to avoid division by zero
    
    return points_3d

