import torch

def block_diag_embed(mat):
    """
    Takes a tensor of shape (..., B, M, N) and returns a block diagonal tensor
    of shape (..., B x M, B x N).

    :param torch.Tensor mat: an input tensor with 3 or more dimensions
    :returns torch.Tensor: a block diagonal tensor with dimension `m.dim() - 1`
    """
    # Get the shape of the input tensor
    *batch_dims, B, M, N = mat.shape
    
    # Create an identity matrix of shape (B, B)
    eye = torch.eye(B, dtype=mat.dtype, device=mat.device)
    
    # Reshape the identity matrix to (B, 1, B, 1)
    eye = eye.view(B, 1, B, 1)
    
    # Expand the identity matrix to (B, M, B, N)
    eye = eye.expand(B, M, B, N)
    
    # Multiply the input tensor with the expanded identity matrix
    # This will place each MxN block along the diagonal
    block_diag = mat.unsqueeze(-3) * eye
    
    # Reshape the result to (..., B * M, B * N)
    block_diag = block_diag.reshape(*batch_dims, B * M, B * N)
    
    return block_diag