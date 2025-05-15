def diff(u, t, order=1, shape_check=True):
    r"""The derivative of a variable with respect to another. ``diff`` defaults to the behaviour of ``safe_diff``.

    :param u: The :math:`u` in :math:`\displaystyle\frac{\partial u}{\partial t}`.
    :type u: `torch.Tensor`
    :param t: The :math:`t` in :math:`\displaystyle\frac{\partial u}{\partial t}`.
    :type t: `torch.Tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :param shape_check: Whether to perform shape checking or not, defaults to True (since v0.2.0).
    :type shape_check: bool
    :returns: The derivative evaluated at t.
    :rtype: `torch.Tensor`
    """

    if shape_check:
        return safe_diff(u, t, order=order)
    else:
        return unsafe_diff(u, t, order=order)