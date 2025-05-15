import torch

def convert_points_to_homogeneous(points):
    """
    Convert Euclidean points to homogeneous coordinates.

    Args:
        points (torch.Tensor): A tensor of shape (*, N, D) representing Euclidean points.

    Returns:
        torch.Tensor: A tensor of shape (*, N, D+1) representing points in homogeneous coordinates.

    Raises:
        TypeError: If the input is not a tensor.
        ValueError: If the input tensor has fewer than two dimensions.
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    
    if points.dim() < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    
    # Get the shape of the input tensor
    *batch_dims, N, D = points.shape
    
    # Create a tensor of ones with the same batch dimensions and N points
    ones = torch.ones(*batch_dims, N, 1, dtype=points.dtype, device=points.device)
    
    # Concatenate the ones to the last dimension of the points tensor
    homogeneous_points = torch.cat([points, ones], dim=-1)
    
    return homogeneous_points

def _convert_affinematrix_to_homography_impl(affine_matrix):
    """
    Convert an affine matrix to a homography matrix.

    Args:
        affine_matrix (torch.Tensor): A tensor of shape (*, D, D+1) representing affine matrices.

    Returns:
        torch.Tensor: A tensor of shape (*, D+1, D+1) representing homography matrices.
    """
    if not isinstance(affine_matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    
    if affine_matrix.dim() < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    
    # Get the shape of the input tensor
    *batch_dims, D, D_plus_1 = affine_matrix.shape
    
    if D_plus_1 != D + 1:
        raise ValueError("The last dimension of the affine matrix must be D+1.")
    
    # Create an identity matrix of shape (*, 1, D+1)
    identity_row = torch.zeros(*batch_dims, 1, D_plus_1, dtype=affine_matrix.dtype, device=affine_matrix.device)
    identity_row[..., -1] = 1  # Set the last element to 1
    
    # Concatenate the identity row to the affine matrix
    homography_matrix = torch.cat([affine_matrix, identity_row], dim=-2)
    
    return homography_matrix