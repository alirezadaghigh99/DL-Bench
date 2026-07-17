"""Module containing numerical functionalities for SfM."""
import torch
from kornia.core import stack, zeros_like

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    """Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.
    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]
    zeros = zeros_like(x0)
    cross_product_matrix_flat = stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)

def matrix_cofactor_tensor(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the cofactor matrix for a given tensor of matrices.

    Args:
        matrix: The input tensor of matrices in the shape :math:`(*, 3, 3)`.

    Returns:
        The tensor containing the cofactor matrices with shape :math:`(*, 3, 3)`.
    """
    if not matrix.shape[-2:] == (3, 3):
        raise AssertionError(matrix.shape)

    determinant = torch.det(matrix)
    if bool(torch.all(determinant == 0)):
        raise Exception('The determinant of every input matrix is 0.')

    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d

    cofactor_flat = stack([c00, c01, c02, c10, c11, c12, c20, c21, c22], dim=-1)
    shape_ = matrix.shape[:-2] + (3, 3)
    return cofactor_flat.view(*shape_)
