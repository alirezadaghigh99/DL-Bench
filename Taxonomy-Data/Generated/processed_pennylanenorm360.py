def norm(tensor, interface='scipy', axis=None, **kwargs):
    if interface == 'jax':
        from jax.numpy import linalg as jax_linalg
        return jax_linalg.norm(tensor, axis=axis, **kwargs)
    
    elif interface == 'tensorflow':
        import tensorflow as tf
        return tf.norm(tensor, axis=axis, **kwargs)
    
    elif interface == 'torch':
        import torch
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            return torch.linalg.norm(tensor, dim=axis, **kwargs)
        else:
            return torch.linalg.norm(tensor, **kwargs)
    
    elif interface == 'autograd':
        import autograd.numpy as anp
        if 'ord' in kwargs and kwargs['ord'] is None:
            return _flat_autograd_norm(tensor, **kwargs)
        else:
            return anp.linalg.norm(tensor, axis=axis, **kwargs)
    
    else:  # default to 'scipy'
        from scipy.linalg import norm as scipy_norm
        return scipy_norm(tensor, axis=axis, **kwargs)

def _flat_autograd_norm(tensor, **kwargs):
    import autograd.numpy as anp
    flat_tensor = anp.ravel(tensor)
    return anp.sqrt(anp.sum(flat_tensor ** 2, **kwargs))

