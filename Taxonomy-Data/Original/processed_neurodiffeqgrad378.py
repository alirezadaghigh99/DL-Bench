def grad(u, *xs):
    r"""Gradient of tensor u with respect to a tuple of tensors xs.
    Given :math:`u` and :math:`x_1`, ..., :math:`x_n`, the function returns
    :math:`\frac{\partial u}{\partial x_1}`, ..., :math:`\frac{\partial u}{\partial x_n}`

    :param u: The :math:`u` described above.
    :type u: `torch.Tensor`
    :param xs: The sequence of :math:`x_i` described above.
    :type xs: `torch.Tensor`
    :return: A tuple of :math:`\frac{\partial u}{\partial x_1}`, ..., :math:`\frac{\partial u}{\partial x_n}`
    :rtype: List[`torch.Tensor`]
    """
    grads = []
    for x, g in zip(xs, autograd.grad(u, xs, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)):
        if g is None:
            grads.append(torch.zeros_like(x, requires_grad=True))
        else:
            grads.append(g.requires_grad_(True))
    return grads