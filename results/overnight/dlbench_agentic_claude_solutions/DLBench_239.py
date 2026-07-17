import math
import warnings
from contextlib import contextmanager
import torch

@contextmanager
def ignore_torch_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'torch.triangular_solve is deprecated')
        yield

def rinverse(M, sym=False):
    """Matrix inversion of rightmost dimensions (batched).

    For 1, 2, and 3 dimensions this uses the formulae.
    For larger matrices, it uses blockwise inversion to reduce to
    smaller matrices.
    """
    assert M.shape[-1] == M.shape[-2]
    n = M.shape[-1]
    if n == 1:
        return 1. / M
    elif n == 2:
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
        inv = torch.empty_like(M)
        inv[..., 0, 0] = M[..., 1, 1]
        inv[..., 1, 1] = M[..., 0, 0]
        inv[..., 0, 1] = -M[..., 0, 1]
        inv[..., 1, 0] = -M[..., 1, 0]
        return inv / det.unsqueeze(-1).unsqueeze(-1)
    elif n == 3:
        return inv3d(M, sym=sym)
    else:
        # Blockwise inversion via the Schur complement, recursing down to
        # the closed-form cases above.
        n_a = (n + 1) // 2
        A = M[..., :n_a, :n_a]
        B = M[..., :n_a, n_a:]
        C = M[..., n_a:, :n_a]
        D = M[..., n_a:, n_a:]

        A_inv = rinverse(A, sym=sym)
        schur = D - C @ A_inv @ B
        schur_inv = rinverse(schur, sym=sym)

        top_left = A_inv + A_inv @ B @ schur_inv @ C @ A_inv
        top_right = -A_inv @ B @ schur_inv
        bottom_left = -schur_inv @ C @ A_inv
        bottom_right = schur_inv

        top = torch.cat([top_left, top_right], dim=-1)
        bottom = torch.cat([bottom_left, bottom_right], dim=-1)
        return torch.cat([top, bottom], dim=-2)

def determinant_3d(H):
    """
    Returns the determinants of a batched 3-D matrix
    """
    detH = H[..., 0, 0] * (H[..., 1, 1] * H[..., 2, 2] - H[..., 2, 1] * H[..., 1, 2]) + H[..., 0, 1] * (H[..., 1, 2] * H[..., 2, 0] - H[..., 1, 0] * H[..., 2, 2]) + H[..., 0, 2] * (H[..., 1, 0] * H[..., 2, 1] - H[..., 2, 0] * H[..., 1, 1])
    return detH

def eig_3d(H):
    """
    Returns the eigenvalues of a symmetric batched 3-D matrix
    """
    p1 = H[..., 0, 1].pow(2) + H[..., 0, 2].pow(2) + H[..., 1, 2].pow(2)
    q = (H[..., 0, 0] + H[..., 1, 1] + H[..., 2, 2]) / 3
    p2 = (H[..., 0, 0] - q).pow(2) + (H[..., 1, 1] - q).pow(2) + (H[..., 2, 2] - q).pow(2) + 2 * p1
    p = torch.sqrt(p2 / 6)
    B = (1 / p).unsqueeze(-1).unsqueeze(-1) * (H - q.unsqueeze(-1).unsqueeze(-1) * torch.eye(3))
    r = determinant_3d(B) / 2
    phi = (r.acos() / 3).unsqueeze(-1).unsqueeze(-1).expand(r.shape + (3, 3)).clone()
    phi[r < -1 + 1e-06] = math.pi / 3
    phi[r > 1 - 1e-06] = 0.0
    eig1 = q + 2 * p * torch.cos(phi[..., 0, 0])
    eig2 = q + 2 * p * torch.cos(phi[..., 0, 0] + 2 * math.pi / 3)
    eig3 = 3 * q - eig1 - eig2
    return (eig2, eig3, eig1)

def inv3d(H, sym=False):
    """
    Calculates the inverse of a batched 3-D matrix
    """
    detH = determinant_3d(H)
    Hinv = torch.empty_like(H)
    Hinv[..., 0, 0] = H[..., 1, 1] * H[..., 2, 2] - H[..., 1, 2] * H[..., 2, 1]
    Hinv[..., 1, 1] = H[..., 0, 0] * H[..., 2, 2] - H[..., 0, 2] * H[..., 2, 0]
    Hinv[..., 2, 2] = H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] * H[..., 1, 0]
    Hinv[..., 0, 1] = H[..., 0, 2] * H[..., 2, 1] - H[..., 0, 1] * H[..., 2, 2]
    Hinv[..., 0, 2] = H[..., 0, 1] * H[..., 1, 2] - H[..., 0, 2] * H[..., 1, 1]
    Hinv[..., 1, 2] = H[..., 0, 2] * H[..., 1, 0] - H[..., 0, 0] * H[..., 1, 2]
    if sym:
        Hinv[..., 1, 0] = Hinv[..., 0, 1]
        Hinv[..., 2, 0] = Hinv[..., 0, 2]
        Hinv[..., 2, 1] = Hinv[..., 1, 2]
    else:
        Hinv[..., 1, 0] = H[..., 2, 0] * H[..., 1, 2] - H[..., 1, 0] * H[..., 2, 2]
        Hinv[..., 2, 0] = H[..., 1, 0] * H[..., 2, 1] - H[..., 2, 0] * H[..., 1, 1]
        Hinv[..., 2, 1] = H[..., 2, 0] * H[..., 0, 1] - H[..., 0, 0] * H[..., 2, 1]
    Hinv = Hinv / detH.unsqueeze(-1).unsqueeze(-1)
    return Hinv
