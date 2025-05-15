def change_basis_real_to_complex(
        k: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None) -> torch.Tensor:
    r"""Construct a transformation matrix to change the basis from real to complex spherical harmonics.

    This function constructs a transformation matrix Q that converts real spherical
    harmonics into complex spherical harmonics.
    It operates on the basis functions $Y_{\ell m}$ and $Y_{\ell}^{m}$, and accounts
    for the relationship between the real and complex forms of these harmonics
    as defined in the provided mathematical expressions.

    The resulting transformation matrix Q is used to change the basis of vectors or tensors of real spherical harmonics to
    their complex counterparts.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    dtype : torch.dtype, optional
        The data type for the output tensor. If not provided, the
        function will infer it. Default is None.
    device : torch.device, optional
        The device where the output tensor will be placed. If not provided,
        the function will use the default device. Default is None.

    Returns
    -------
    torch.Tensor
        A transformation matrix Q that changes the basis from real to complex spherical harmonics.

    Notes
    -----
    Spherical harmonics Y_l^m are a family of functions that are defined on the surface of a
    unit sphere. They are used to represent various physical and mathematical phenomena that
    exhibit spherical symmetry. The indices l and m represent the degree and order of the
    spherical harmonics, respectively.

    The conversion from real to complex spherical harmonics is achieved by applying specific
    transformation coefficients to the real-valued harmonics. These coefficients are derived
    from the properties of spherical harmonics.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    Examples
    --------
    # The transformation matrix generated is used to change the basis of a vector of
    # real spherical harmonics with representation index 1 to complex spherical harmonics.
    >>> change_basis_real_to_complex(1)
    tensor([[-0.7071+0.0000j,  0.0000+0.0000j,  0.0000-0.7071j],
            [ 0.0000+0.0000j,  0.0000-1.0000j,  0.0000+0.0000j],
            [-0.7071+0.0000j,  0.0000+0.0000j,  0.0000+0.7071j]])
    """
    q = torch.zeros((2 * k + 1, 2 * k + 1), dtype=torch.complex128)

    # Construct the transformation matrix Q for m in range(-k, 0)
    for m in range(-k, 0):
        q[k + m, k + abs(m)] = 1 / 2**0.5
        q[k + m, k - abs(m)] = complex(-1j / 2**0.5)  # type: ignore

    # Set the diagonal elements for m = 0
    q[k, k] = 1

    # Construct the transformation matrix Q for m in range(1, k + 1)
    for m in range(1, k + 1):
        q[k + m, k + abs(m)] = (-1)**m / 2**0.5
        q[k + m, k - abs(m)] = complex(1j * (-1)**m / 2**0.5)  # type: ignore

    # Apply the factor of (-1j)**k to make the Clebsch-Gordan coefficients real
    q = (-1j)**k * q

    # Handle dtype and device options
    if dtype is None:
        default_type = torch.empty(0).dtype
        if default_type == torch.float32:
            dtype = torch.complex64
        elif default_type == torch.float64:
            dtype = torch.complex128
    if device is None:
        device = torch.empty(0).device

    # Ensure the tensor is contiguous and on the specified device
    return q.to(
        dtype=dtype,
        device=device,
        copy=True,
        memory_format=torch.contiguous_format)  # type: ignore[call-overload]