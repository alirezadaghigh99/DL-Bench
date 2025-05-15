import torch
import math

def dct(x, dim=-1):
    """
    Compute the Discrete Cosine Transform of type II, scaled to be orthonormal.

    :param Tensor x: The input signal.
    :param int dim: Dimension along which to compute DCT.
    :rtype: Tensor
    """
    N = x.size(dim)
    
    # Compute weights for orthonormal scaling
    scale = torch.sqrt(2.0 / N) * torch.ones(N, device=x.device)
    scale[0] = scale[0] / math.sqrt(2.0)
    
    # Compute the DCT using FFT
    x = torch.fft.fft(torch.cat([x, x.flip(dims=[dim])], dim=dim), dim=dim)
    x = x.real
    
    # Select the first N elements and apply the scaling
    dct_result = scale * x.narrow(dim, 0, N)
    
    return dct_result

