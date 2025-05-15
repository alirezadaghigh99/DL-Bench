def dct(x, dim=-1):
    """
    Discrete cosine transform of type II, scaled to be orthonormal.

    This is the inverse of :func:`idct_ii` , and is equivalent to
    :func:`scipy.fftpack.dct` with ``norm="ortho"``.

    :param Tensor x: The input signal.
    :param int dim: Dimension along which to compute DCT.
    :rtype: Tensor
    """
    if dim >= 0:
        dim -= x.dim()
    if dim != -1:
        y = x.reshape(x.shape[: dim + 1] + (-1,)).transpose(-1, -2)
        return dct(y).transpose(-1, -2).reshape(x.shape)

    # Ref: http://fourier.eng.hmc.edu/e161/lectures/dct/node2.html
    N = x.size(-1)
    # Step 1
    y = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    # Step 2
    Y = rfft(y, n=N)
    # Step 3
    coef_real = torch.cos(
        torch.linspace(0, 0.5 * math.pi, N + 1, dtype=x.dtype, device=x.device)
    )
    M = Y.size(-1)
    coef = torch.stack([coef_real[:M], -coef_real[-M:].flip(-1)], dim=-1)
    X = as_complex(coef) * Y
    # NB: if we use the full-length version Y_full = fft(y, n=N), then
    # the real part of the later half of X will be the flip
    # of the negative of the imaginary part of the first half
    X = torch.cat([X.real, -X.imag[..., 1 : (N - M + 1)].flip(-1)], dim=-1)
    # orthogonalize
    scale = torch.cat(
        [x.new_tensor([math.sqrt(N)]), x.new_full((N - 1,), math.sqrt(0.5 * N))]
    )
    return X / scale