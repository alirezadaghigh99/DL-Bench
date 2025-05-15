def intersect_interval(interval1: Tuple[float, float],
                       interval2: Tuple[float, float]) -> Tuple[float, float]:
    """Computes the intersection of two intervals.

    Parameters
    ----------
    interval1: Tuple[float, float]
        Should be `(x1_min, x1_max)`
    interval2: Tuple[float, float]
        Should be `(x2_min, x2_max)`

    Returns
    -------
    x_intersect: Tuple[float, float]
        Should be the intersection. If the intersection is empty returns
        `(0, 0)` to represent the empty set. Otherwise is `(max(x1_min,
        x2_min), min(x1_max, x2_max))`.
    """
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    if x1_max < x2_min:
        # If interval1 < interval2 entirely
        return (0, 0)
    elif x2_max < x1_min:
        # If interval2 < interval1 entirely
        return (0, 0)
    x_min = max(x1_min, x2_min)
    x_max = min(x1_max, x2_max)
    return (x_min, x_max)