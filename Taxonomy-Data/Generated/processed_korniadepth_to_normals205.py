import torch

def depth_to_normals(depth, camera_matrix, normalize_points=False):
    # Check input types
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a Tensor")
    
    # Check input shapes
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape (B, 1, H, W)")
    if camera_matrix.ndim != 3 or camera_matrix.shape[1:] != (3, 3):
        raise ValueError("camera_matrix must have shape (B, 3, 3)")
    
    B, _, H, W = depth.shape
    
    # Create a meshgrid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
    y = y.expand(B, -1, -1)
    x = x.expand(B, -1, -1)
    
    # Unpack camera intrinsics
    fx = camera_matrix[:, 0, 0]
    fy = camera_matrix[:, 1, 1]
    cx = camera_matrix[:, 0, 2]
    cy = camera_matrix[:, 1, 2]
    
    # Convert depth to 3D points in camera space
    z = depth[:, 0, :, :]
    x3d = (x - cx.view(-1, 1, 1)) * z / fx.view(-1, 1, 1)
    y3d = (y - cy.view(-1, 1, 1)) * z / fy.view(-1, 1, 1)
    points_3d = torch.stack((x3d, y3d, z), dim=1)
    
    if normalize_points:
        points_3d = points_3d / torch.norm(points_3d, dim=1, keepdim=True)
    
    # Compute gradients
    dzdx = torch.gradient(z, dim=2)[0]
    dzdy = torch.gradient(z, dim=1)[0]
    
    # Compute normals using cross product
    normals = torch.cross(
        torch.stack((torch.ones_like(dzdx), torch.zeros_like(dzdx), dzdx), dim=1),
        torch.stack((torch.zeros_like(dzdy), torch.ones_like(dzdy), dzdy), dim=1),
        dim=1
    )
    
    # Normalize the normals
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    
    return normals

