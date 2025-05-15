def block_diag_embed(mat):
    """
    Takes a tensor of shape (..., B, M, N) and returns a block diagonal tensor
    of shape (..., B x M, B x N).

    :param torch.Tensor mat: an input tensor with 3 or more dimensions
    :returns torch.Tensor: a block diagonal tensor with dimension `m.dim() - 1`
    """
    assert mat.dim() > 2, "Input to block_diag() must be of dimension 3 or higher"
    B, M, N = mat.shape[-3:]
    eye = torch.eye(B, dtype=mat.dtype, device=mat.device).reshape(B, 1, B, 1)
    return (mat.unsqueeze(-2) * eye).reshape(mat.shape[:-3] + (B * M, B * N))