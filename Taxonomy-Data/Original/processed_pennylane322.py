def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    """Wrapper around np.allclose, allowing tensors ``a`` and ``b``
    to differ in type"""
    try:
        # Some frameworks may provide their own allclose implementation.
        # Try and use it if available.
        res = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
    except (TypeError, AttributeError, ImportError, RuntimeError):
        # Otherwise, convert the input to NumPy arrays.
        #
        # TODO: replace this with a bespoke, framework agnostic
        # low-level implementation to avoid the NumPy conversion:
        #
        #    np.abs(a - b) <= atol + rtol * np.abs(b)
        #
        t1 = ar.to_numpy(a)
        t2 = ar.to_numpy(b)
        res = np.allclose(t1, t2, rtol=rtol, atol=atol, **kwargs)

    return res