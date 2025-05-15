def _calculate_knots(lengths, lower, upper):
    """
    Given a tensor of unscaled bin lengths that sum to 1, plus the lower and upper
    limits, returns the shifted and scaled lengths plus knot positions
    """

    # Cumulative widths gives x (y for inverse) position of knots
    knots = torch.cumsum(lengths, dim=-1)

    # Pad left of last dimension with 1 zero to compensate for dim lost to cumsum
    knots = F.pad(knots, pad=(1, 0), mode="constant", value=0.0)

    # Translate [0,1] knot points to [-B, B]
    knots = (upper - lower) * knots + lower

    # Convert the knot points back to lengths
    # NOTE: Are following two lines a necessary fix for accumulation (round-off) error?
    knots[..., 0] = lower
    knots[..., -1] = upper
    lengths = knots[..., 1:] - knots[..., :-1]

    return lengths, knots