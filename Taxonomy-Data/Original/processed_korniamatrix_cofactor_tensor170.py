def matrix_cofactor_tensor(matrix: torch.Tensor) -> torch.Tensor:
    """Cofactor matrix, refer to the numpy doc.

    Args:
        matrix: The input matrix in the shape :math:`(*, 3, 3)`.
    """
    det = torch.det(matrix)
    singular_mask = det != 0
    if singular_mask.sum() != 0:
        # B, 3, 3
        cofactor = torch.linalg.inv(matrix[singular_mask]).transpose(-2, -1) * det[:, None, None]
        # return cofactor matrix of the given matrix
        returned_cofactor = torch.zeros_like(matrix)
        returned_cofactor[singular_mask] = cofactor
        return returned_cofactor
    else:
        raise Exception("all singular matrices")