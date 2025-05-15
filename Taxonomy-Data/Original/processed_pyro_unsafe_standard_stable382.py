def _unsafe_standard_stable(alpha, beta, V, W, coords):
    # Implements a noisily reparametrized version of the sampler
    # Chambers-Mallows-Stuck method as corrected by Weron [1,3] and simplified
    # by Nolan [4]. This will fail if alpha is close to 1.

    # Differentiably transform noise via parameters.
    assert V.shape == W.shape
    inv_alpha = alpha.reciprocal()
    half_pi = math.pi / 2
    eps = torch.finfo(V.dtype).eps
    # make V belong to the open interval (-pi/2, pi/2)
    V = V.clamp(min=2 * eps - half_pi, max=half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * ha.tan()
    # +/- `ha` term to keep the precision of alpha * (V + half_pi) when V ~ -half_pi
    v = b.atan() - ha + alpha * (V + half_pi)
    Z = (
        v.sin()
        / ((1 + b * b).rsqrt() * V.cos()).pow(inv_alpha)
        * ((v - V).cos().clamp(min=eps) / W).pow(inv_alpha - 1)
    )
    Z.data[Z.data != Z.data] = 0  # drop occasional NANs

    # Optionally convert to Nolan's parametrization S^0 where samples depend
    # continuously on (alpha,beta), allowing interpolation around the hole at
    # alpha=1.
    if coords == "S0":
        return Z - b
    elif coords == "S":
        return Z
    else:
        raise ValueError("Unknown coords: {}".format(coords))