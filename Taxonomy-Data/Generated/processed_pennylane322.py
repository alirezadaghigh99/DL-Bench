import numpy as np

def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    try:
        # Attempt to use np.allclose directly
        res = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
    except (TypeError, AttributeError, ImportError, RuntimeError):
        # Convert inputs to NumPy arrays if they are not already
        a = np.asarray(a)
        b = np.asarray(b)
        res = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
    
    return res