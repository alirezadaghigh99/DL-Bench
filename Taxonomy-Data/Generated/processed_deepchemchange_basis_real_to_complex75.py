import torch
import math

def change_basis_real_to_complex(k, dtype=None, device=None):
    """
    Constructs a transformation matrix Q that converts real spherical
    harmonics into complex spherical harmonics.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    dtype : torch.dtype, optional
        The data type for the output tensor. If not provided, the
        function will infer it. Default is None.
    device : torch.device, optional
        The device where the output tensor will be placed. If not provided,
        the function will use the default device. Default is None.

    Returns
    -------
    torch.Tensor
        A transformation matrix Q that changes the basis from real to complex spherical harmonics.
    """
    # Determine the size of the transformation matrix
    size = 2 * k + 1

    # Initialize the transformation matrix Q
    Q = torch.zeros((size, size), dtype=dtype, device=device)

    # Fill the transformation matrix based on the relationship between real and complex harmonics
    for m in range(-k, k + 1):
        if m < 0:
            # Real part: Y_l^m = sqrt(2) * (-1)^m * Im(Y_l^|m|)
            Q[m + k, k + m] = 1j / math.sqrt(2)
            Q[m + k, k - m] = (-1)**m / math.sqrt(2)
        elif m == 0:
            # Real part: Y_l^0 = Y_l^0
            Q[m + k, k] = 1.0
        else:
            # Real part: Y_l^m = sqrt(2) * (-1)^m * Re(Y_l^m)
            Q[m + k, k + m] = 1 / math.sqrt(2)
            Q[m + k, k - m] = 1j * (-1)**m / math.sqrt(2)

    return Q

