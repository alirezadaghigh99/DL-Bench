def norm(tensor, like=None, **kwargs):
    """Compute the norm of a tensor in each interface."""
    if like == "jax":
        from jax.numpy.linalg import norm

    elif like == "tensorflow":
        from tensorflow import norm

    elif like == "torch":
        from torch.linalg import norm

        if "axis" in kwargs:
            axis_val = kwargs.pop("axis")
            kwargs["dim"] = axis_val

    elif (
        like == "autograd" and kwargs.get("ord", None) is None and kwargs.get("axis", None) is None
    ):
        norm = _flat_autograd_norm

    else:
        from scipy.linalg import norm

    return norm(tensor, **kwargs)