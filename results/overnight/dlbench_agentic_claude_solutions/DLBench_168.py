"""Module containing the functionalities for computing the Fundamental Matrix."""
from typing import Literal, Optional, Tuple
import torch
from kornia.core import Tensor, concatenate, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points
from kornia.geometry.solvers import solve_cubic
from kornia.utils.helpers import _torch_svd_cast, safe_inverse_with_mask

def normalize_points(points: Tensor, eps: float=1e-08) -> Tuple[Tensor, Tensor]:
    """Normalizes points (isotropic).

    Computes the transformation matrix such that the two principal moments of the set of points
    are equal to unity, forming an approximately symmetric circular cloud of points of radius 1
    about the origin. Reference: Hartley/Zisserman 4.4.4 pag.107

    This operation is an essential step before applying the DLT algorithm in order to consider
    the result as optimal.

    Args:
       points: Tensor containing the points to be normalized with shape :math:`(B, N, 2)`.
       eps: epsilon value to avoid numerical instabilities.

    Returns:
       tuple containing the normalized points in the shape :math:`(B, N, 2)` and the transformation matrix
       in the shape :math:`(B, 3, 3)`.
    """
    if len(points.shape) != 3:
        raise AssertionError(points.shape)
    if points.shape[-1] != 2:
        raise AssertionError(points.shape)
    x_mean = torch.mean(points, dim=1, keepdim=True)
    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)
    scale = torch.sqrt(torch.tensor(2.0)) / (scale + eps)
    (ones, zeros) = (ones_like(scale), torch.zeros_like(scale))
    transform = stack([scale, zeros, -scale * x_mean[..., 0, 0], zeros, scale, -scale * x_mean[..., 0, 1], zeros, zeros, ones], dim=-1)
    transform = transform.view(-1, 3, 3)
    points_norm = transform_points(transform, points)
    return (points_norm, transform)

def normalize_transformation(M: Tensor, eps: float=1e-08) -> Tensor:
    """Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.
    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: Tensor = M[..., -1:, -1:]
    return where(norm_val.abs() > eps, M / (norm_val + eps), M)

def run_7point(points1: Tensor, points2: Tensor) -> Tensor:
    """Compute the fundamental matrix using the 7-point algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, 7, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, 7, 2)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3)`, where `m` number of fundamental matrix.
    """
    KORNIA_CHECK_SHAPE(points1, ['B', 'N', '2'])
    KORNIA_CHECK_SHAPE(points2, ['B', 'N', '2'])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    if points1.shape[1] != 7:
        raise AssertionError(points1.shape)
    (points1_norm, transform1) = normalize_points(points1)
    (points2_norm, transform2) = normalize_points(points2)
    (x1, y1) = torch.chunk(points1_norm, dim=-1, chunks=2)
    (x2, y2) = torch.chunk(points2_norm, dim=-1, chunks=2)
    ones = ones_like(x1)
    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)

    # X has a rank of 7, so its null space is 2-dimensional: any linear combination of the two
    # null-space vectors f1, f2 (last two columns of V) satisfies the linear system.
    (_, _, v) = _torch_svd_cast(X)
    f1 = v[..., 7].view(-1, 3, 3)
    f2 = v[..., 8].view(-1, 3, 3)

    # The valid fundamental matrix is F(lambda) = lambda * f1 + (1 - lambda) * f2 = lambda * g + f2,
    # with g = f1 - f2. Imposing det(F(lambda)) = 0 yields a cubic polynomial in lambda whose
    # coefficients are obtained from the cofactor expansion of det(g + lambda * ... ) (port of the
    # classic OpenCV 7-point solver, see modules/calib3d/src/fundam.cpp `sevenPoint`).
    g = f1 - f2
    (g0, g1, g2, g3, g4, g5, g6, g7, g8) = (g[..., 0, 0], g[..., 0, 1], g[..., 0, 2], g[..., 1, 0], g[..., 1, 1], g[..., 1, 2], g[..., 2, 0], g[..., 2, 1], g[..., 2, 2])
    (h0, h1, h2, h3, h4, h5, h6, h7, h8) = (f2[..., 0, 0], f2[..., 0, 1], f2[..., 0, 2], f2[..., 1, 0], f2[..., 1, 1], f2[..., 1, 2], f2[..., 2, 0], f2[..., 2, 1], f2[..., 2, 2])

    t0 = h4 * h8 - h5 * h7
    t1 = h3 * h8 - h5 * h6
    t2 = h3 * h7 - h4 * h6
    c3 = h0 * t0 - h1 * t1 + h2 * t2
    c2 = (
        g0 * t0 - g1 * t1 + g2 * t2
        - g3 * (h1 * h8 - h2 * h7) + g4 * (h0 * h8 - h2 * h6) - g5 * (h0 * h7 - h1 * h6)
        + g6 * (h1 * h5 - h2 * h4) - g7 * (h0 * h5 - h2 * h3) + g8 * (h0 * h4 - h1 * h3)
    )

    u0 = g4 * g8 - g5 * g7
    u1 = g3 * g8 - g5 * g6
    u2 = g3 * g7 - g4 * g6
    c0 = g0 * u0 - g1 * u1 + g2 * u2
    c1 = (
        h0 * u0 - h1 * u1 + h2 * u2
        - h3 * (g1 * g8 - g2 * g7) + h4 * (g0 * g8 - g2 * g6) - h5 * (g0 * g7 - g1 * g6)
        + h6 * (g1 * g5 - g2 * g4) - h7 * (g0 * g5 - g2 * g3) + h8 * (g0 * g4 - g1 * g3)
    )

    coeffs = stack([c0, c1, c2, c3], dim=-1)
    roots = solve_cubic(coeffs)

    # solve_cubic zero-pads unused root slots; keep only the slots that are actually populated
    # for at least one sample in the batch, so `m` reflects the real number of candidates.
    if torch.any(roots[:, 2] != 0):
        num_roots = 3
    elif torch.any(roots[:, 1] != 0):
        num_roots = 2
    else:
        num_roots = 1

    eps = 1e-8
    g_last = g[..., 2, 2]
    f2_last = f2[..., 2, 2]
    F_candidates = []
    for i in range(num_roots):
        r = roots[:, i]
        s = g_last * r + f2_last
        mask_s = s.abs() > eps
        s_safe = where(mask_s, s, ones_like(s))
        mu = where(mask_s, 1.0 / s_safe, ones_like(s))
        lam = where(mask_s, r * mu, r)
        F_i = lam.view(-1, 1, 1) * g + mu.view(-1, 1, 1) * f2
        F_est = transform2.transpose(-2, -1) @ (F_i @ transform1)
        F_candidates.append(normalize_transformation(F_est))
    return concatenate(F_candidates, dim=1)

def run_8point(points1: Tensor, points2: Tensor, weights: Optional[Tensor]=None) -> Tensor:
    """Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(points1, ['B', 'N', '2'])
    KORNIA_CHECK_SHAPE(points2, ['B', 'N', '2'])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    if points1.shape[1] < 8:
        raise AssertionError(points1.shape)
    if weights is not None:
        KORNIA_CHECK_SHAPE(weights, ['B', 'N'])
        if not weights.shape[1] == points1.shape[1]:
            raise AssertionError(weights.shape)
    (points1_norm, transform1) = normalize_points(points1)
    (points2_norm, transform2) = normalize_points(points2)
    (x1, y1) = torch.chunk(points1_norm, dim=-1, chunks=2)
    (x2, y2) = torch.chunk(points2_norm, dim=-1, chunks=2)
    ones = ones_like(x1)
    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)
    if weights is None:
        X = X.transpose(-2, -1) @ X
    else:
        w_diag = torch.diag_embed(weights)
        X = X.transpose(-2, -1) @ w_diag @ X
    (_, _, V) = _torch_svd_cast(X)
    F_mat = V[..., -1].view(-1, 3, 3)
    (U, S, V) = _torch_svd_cast(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype)
    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)
    return normalize_transformation(F_est)

def find_fundamental(points1: Tensor, points2: Tensor, weights: Optional[Tensor]=None, method: Literal['8POINT', '7POINT']='8POINT') -> Tensor:
    """
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3)`, where `m` number of fundamental matrix.

    Raises:
        ValueError: If an invalid method is provided.

    """
    if method.upper() == '7POINT':
        result = run_7point(points1, points2)
    elif method.upper() == '8POINT':
        result = run_8point(points1, points2, weights)
    else:
        raise ValueError(f"Invalid method: {method}. Supported methods are '7POINT' and '8POINT'.")
    return result

def compute_correspond_epilines(points: Tensor, F_mat: Tensor) -> Tensor:
    """Compute the corresponding epipolar line for a given set of points.

    Args:
        points: tensor containing the set of points to project in the shape of :math:`(*, N, 2)` or :math:`(*, N, 3)`.
        F_mat: the fundamental to use for projection the points in the shape of :math:`(*, 3, 3)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing a vector of the epipolar
        lines corresponding to the points to the other image. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.
    """
    KORNIA_CHECK_SHAPE(points, ['*', 'N', 'DIM'])
    if points.shape[-1] == 2:
        points_h: Tensor = convert_points_to_homogeneous(points)
    elif points.shape[-1] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    KORNIA_CHECK_SHAPE(F_mat, ['*', '3', '3'])
    points_h = torch.transpose(points_h, dim0=-2, dim1=-1)
    (a, b, c) = torch.chunk(F_mat @ points_h, dim=-2, chunks=3)
    nu: Tensor = a * a + b * b
    nu = where(nu > 0.0, 1.0 / torch.sqrt(nu), torch.ones_like(nu))
    line = torch.cat([a * nu, b * nu, c * nu], dim=-2)
    return torch.transpose(line, dim0=-2, dim1=-1)

def get_perpendicular(lines: Tensor, points: Tensor) -> Tensor:
    """Compute the perpendicular to a line, through the point.

    Args:
        lines: tensor containing the set of lines :math:`(*, N, 3)`.
        points:  tensor containing the set of points :math:`(*, N, 2)`.

    Returns:
        a tensor with shape :math:`(*, N, 3)` containing a vector of the epipolar
        perpendicular lines. Each line is described as
        :math:`ax + by + c = 0` and encoding the vectors as :math:`(a, b, c)`.
    """
    KORNIA_CHECK_SHAPE(lines, ['*', 'N', '3'])
    KORNIA_CHECK_SHAPE(points, ['*', 'N', 'two'])
    if points.shape[2] == 2:
        points_h: Tensor = convert_points_to_homogeneous(points)
    elif points.shape[2] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    infinity_point = lines * torch.tensor([1, 1, 0], dtype=lines.dtype, device=lines.device).view(1, 1, 3)
    perp: Tensor = points_h.cross(infinity_point, dim=2)
    return perp

def get_closest_point_on_epipolar_line(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    """Return closest point on the epipolar line to the correspondence, given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.

    Returns:
        point on epipolar line :math:`(*, N, 2)`.
    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f'Fm type is not a torch.Tensor. Got {type(Fm)}')
    if len(Fm.shape) < 3 or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f'Fm must be a (*, 3, 3) tensor. Got {Fm.shape}')
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    line1in2 = compute_correspond_epilines(pts1, Fm)
    perp = get_perpendicular(line1in2, pts2)
    points1_in_2 = convert_points_from_homogeneous(line1in2.cross(perp, dim=2))
    return points1_in_2

def fundamental_from_essential(E_mat: Tensor, K1: Tensor, K2: Tensor) -> Tensor:
    """Get the Fundamental matrix from Essential and camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Args:
        E_mat: The essential matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(E_mat, ['*', '3', '3'])
    KORNIA_CHECK_SHAPE(K1, ['*', '3', '3'])
    KORNIA_CHECK_SHAPE(K2, ['*', '3', '3'])
    if not len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError
    return safe_inverse_with_mask(K2)[0].transpose(-2, -1) @ E_mat @ safe_inverse_with_mask(K1)[0]

def fundamental_from_projections(P1: Tensor, P2: Tensor) -> Tensor:
    """Get the Fundamental matrix from Projection matrices.

    Args:
        P1: The projection matrix from first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix from second camera with shape :math:`(*, 3, 4)`.

    Returns:
         The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(P1, ['*', '3', '4'])
    KORNIA_CHECK_SHAPE(P2, ['*', '3', '4'])
    if P1.shape[:-2] != P2.shape[:-2]:
        raise AssertionError

    def vstack(x: Tensor, y: Tensor) -> Tensor:
        return concatenate([x, y], dim=-2)
    input_dtype = P1.dtype
    if input_dtype not in (torch.float32, torch.float64):
        P1 = P1.to(torch.float32)
        P2 = P2.to(torch.float32)
    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]
    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]
    (X1Y1, X2Y1, X3Y1) = (vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1))
    (X1Y2, X2Y2, X3Y2) = (vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2))
    (X1Y3, X2Y3, X3Y3) = (vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3))
    F_vec = torch.cat([X1Y1.det().reshape(-1, 1), X2Y1.det().reshape(-1, 1), X3Y1.det().reshape(-1, 1), X1Y2.det().reshape(-1, 1), X2Y2.det().reshape(-1, 1), X3Y2.det().reshape(-1, 1), X1Y3.det().reshape(-1, 1), X2Y3.det().reshape(-1, 1), X3Y3.det().reshape(-1, 1)], dim=1)
    return F_vec.view(*P1.shape[:-2], 3, 3).to(input_dtype)
