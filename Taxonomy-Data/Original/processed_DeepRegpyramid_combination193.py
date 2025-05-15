def pyramid_combination(
    values: list, weight_floor: list, weight_ceil: list
) -> tf.Tensor:
    r"""
    Calculates linear interpolation (a weighted sum) using values of
    hypercube corners in dimension n.

    For example, when num_dimension = len(loc_shape) = num_bits = 3
    values correspond to values at corners of following coordinates

    .. code-block:: python

        [[0, 0, 0], # even
         [0, 0, 1], # odd
         [0, 1, 0], # even
         [0, 1, 1], # odd
         [1, 0, 0], # even
         [1, 0, 1], # odd
         [1, 1, 0], # even
         [1, 1, 1]] # odd

    values[::2] correspond to the corners with last coordinate == 0

    .. code-block:: python

        [[0, 0, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 1, 0]]

    values[1::2] correspond to the corners with last coordinate == 1

    .. code-block:: python

        [[0, 0, 1],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]

    The weights correspond to the floor corners.
    For example, when num_dimension = len(loc_shape) = num_bits = 3,
    weight_floor = [f1, f2, f3] (ignoring the batch dimension).
    weight_ceil = [c1, c2, c3] (ignoring the batch dimension).

    So for corner with coords (x, y, z), x, y, z's values are 0 or 1

    - weight for x = f1 if x = 0 else c1
    - weight for y = f2 if y = 0 else c2
    - weight for z = f3 if z = 0 else c3

    so the weight for (x, y, z) is

    .. code-block:: text

        W_xyz = ((1-x) * f1 + x * c1)
              * ((1-y) * f2 + y * c2)
              * ((1-z) * f3 + z * c3)

    Let

    .. code-block:: text

        W_xy = ((1-x) * f1 + x * c1)
             * ((1-y) * f2 + y * c2)

    Then

    - W_xy0 = W_xy * f3
    - W_xy1 = W_xy * c3

    Similar to W_xyz, denote V_xyz the value at (x, y, z),
    the final sum V equals

    .. code-block:: text

          sum over x,y,z (V_xyz * W_xyz)
        = sum over x,y (V_xy0 * W_xy0 + V_xy1 * W_xy1)
        = sum over x,y (V_xy0 * W_xy * f3 + V_xy1 * W_xy * c3)
        = sum over x,y (V_xy0 * W_xy) * f3 + sum over x,y (V_xy1 * W_xy) * c3

    That's why we call this pyramid combination.
    It calculates the linear interpolation gradually, starting from
    the last dimension.
    The key is that the weight of each corner is the product of the weights
    along each dimension.

    :param values: a list having values on the corner,
                   it has 2**n tensors of shape
                   (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, ch)
                   the order is consistent with get_n_bits_combinations
                   loc_shape is independent from n, aka num_dim
    :param weight_floor: a list having weights of floor points,
                    it has n tensors of shape
                    (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    :param weight_ceil: a list having weights of ceil points,
                    it has n tensors of shape
                    (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    :return: one tensor of the same shape as an element in values
             (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    """
    v_shape = values[0].shape
    wf_shape = weight_floor[0].shape
    wc_shape = weight_ceil[0].shape
    if len(v_shape) != len(wf_shape) or len(v_shape) != len(wc_shape):
        raise ValueError(
            "In pyramid_combination, elements of "
            "values, weight_floor, and weight_ceil should have same dimension. "
            f"value shape = {v_shape}, "
            f"weight_floor = {wf_shape}, "
            f"weight_ceil = {wc_shape}."
        )
    if 2 ** len(weight_floor) != len(values):
        raise ValueError(
            "In pyramid_combination, "
            "num_dim = len(weight_floor), "
            "len(values) must be 2 ** num_dim, "
            f"But len(weight_floor) = {len(weight_floor)}, "
            f"len(values) = {len(values)}"
        )

    if len(weight_floor) == 1:  # one dimension
        return values[0] * weight_floor[0] + values[1] * weight_ceil[0]
    # multi dimension
    values_floor = pyramid_combination(
        values=values[::2],
        weight_floor=weight_floor[:-1],
        weight_ceil=weight_ceil[:-1],
    )
    values_floor = values_floor * weight_floor[-1]
    values_ceil = pyramid_combination(
        values=values[1::2],
        weight_floor=weight_floor[:-1],
        weight_ceil=weight_ceil[:-1],
    )
    values_ceil = values_ceil * weight_ceil[-1]
    return values_floor + values_ceil