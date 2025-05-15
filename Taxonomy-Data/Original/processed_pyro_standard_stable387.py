def _standard_stable(alpha, beta, aux_uniform, aux_exponential, coords):
    """
    Differentiably transform two random variables::

        aux_uniform ~ Uniform(-pi/2, pi/2)
        aux_exponential ~ Exponential(1)

    to a standard ``Stable(alpha, beta)`` random variable.
    """
    # Determine whether a hole workaround is needed.
    with torch.no_grad():
        hole = 1.0
        near_hole = (alpha - hole).abs() <= RADIUS
    if not torch._C._get_tracing_state() and not near_hole.any():
        return _unsafe_standard_stable(
            alpha, beta, aux_uniform, aux_exponential, coords=coords
        )
    if coords == "S":
        # S coords are discontinuous, so interpolate instead in S0 coords.
        Z = _standard_stable(alpha, beta, aux_uniform, aux_exponential, "S0")
        return torch.where(alpha == 1, Z, Z + beta * (math.pi / 2 * alpha).tan())

    # Avoid the hole at alpha=1 by interpolating between pairs
    # of points at hole-RADIUS and hole+RADIUS.
    aux_uniform_ = aux_uniform.unsqueeze(-1)
    aux_exponential_ = aux_exponential.unsqueeze(-1)
    beta_ = beta.unsqueeze(-1)
    alpha_ = alpha.unsqueeze(-1).expand(alpha.shape + (2,)).contiguous()
    with torch.no_grad():
        lower, upper = alpha_.unbind(-1)
        lower.data[near_hole] = hole - RADIUS
        upper.data[near_hole] = hole + RADIUS
        # We don't need to backprop through weights, since we've pretended
        # alpha_ is reparametrized, even though we've clamped some values.
        #               |a - a'|
        # weight = 1 - ----------
        #              2 * RADIUS
        weights = (alpha_ - alpha.unsqueeze(-1)).abs_().mul_(-1 / (2 * RADIUS)).add_(1)
        weights[~near_hole] = 0.5
    pairs = _unsafe_standard_stable(
        alpha_, beta_, aux_uniform_, aux_exponential_, coords=coords
    )
    return (pairs * weights).sum(-1)