def U2(phi, delta):
    r"""Return the matrix representation of the U2 gate.

    .. math::

        U_2(\phi, \delta) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -\exp(i \delta)
        \\ \exp(i \phi) & \exp(i (\phi + \delta)) \end{bmatrix}

    Args:dd
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
    """
    return (
        1
        / math.sqrt(2)
        * math.array(
            [[1.0, -math.exp(delta * 1j)], [math.exp(phi * 1j), math.exp((phi + delta) * 1j)]]
        )
    )