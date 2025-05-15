import torch
from scipy.special import factorial

def wigner_D(k, alpha, beta, gamma):
    """
    Compute the Wigner D matrix for the SO(3) rotation group.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    alpha : torch.Tensor
        Rotation angles (in radians) around the Y axis, applied third.
    beta : torch.Tensor
        Rotation angles (in radians) around the X axis, applied second.
    gamma : torch.Tensor
        Rotation angles (in radians) around the Y axis, applied first.

    Returns
    -------
    torch.Tensor
        The Wigner D matrix of shape (#angles, 2k+1, 2k+1).
    """
    # Ensure angles are tensors
    alpha = torch.tensor(alpha, dtype=torch.float64)
    beta = torch.tensor(beta, dtype=torch.float64)
    gamma = torch.tensor(gamma, dtype=torch.float64)

    # Number of angles
    num_angles = alpha.shape[0]

    # Initialize the Wigner D matrix
    D = torch.zeros((num_angles, 2*k+1, 2*k+1), dtype=torch.complex128)

    # Compute the Wigner D matrix elements
    for m in range(-k, k+1):
        for n in range(-k, k+1):
            # Compute the Wigner small d matrix element
            d_mn = 0
            for s in range(max(0, m-n), min(k+m, k-n)+1):
                term = ((-1)**(m-n+s) *
                        factorial(k+m) * factorial(k-m) * factorial(k+n) * factorial(k-n) /
                        (factorial(k+m-s) * factorial(k-n-s) * factorial(s) * factorial(s+m-n)))
                term *= (torch.cos(beta/2)**(2*k+m-n-2*s) * torch.sin(beta/2)**(2*s+m-n))
                d_mn += term

            # Compute the full Wigner D matrix element
            D[:, m+k, n+k] = torch.exp(-1j * m * alpha) * d_mn * torch.exp(-1j * n * gamma)

    return D

