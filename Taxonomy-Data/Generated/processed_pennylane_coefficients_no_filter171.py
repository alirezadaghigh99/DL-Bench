import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=True):
    if isinstance(degree, int):
        d = degree
        k_values = np.arange(-d, d + 1)
    elif isinstance(degree, tuple) and len(degree) == 2:
        d1, d2 = degree
        k_values = np.array([(k1, k2) for k1 in range(-d1, d1 + 1) for k2 in range(-d2, d2 + 1)])
    else:
        raise ValueError("degree must be an integer or a tuple of two integers")
    
    # Number of sample points for numerical integration
    num_points = 1000
    x = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    dx = x[1] - x[0]  # Step size

    if use_broadcasting:
        # Evaluate the function at all points simultaneously
        fx = f(x)
    else:
        # Evaluate the function at each point individually
        fx = np.array([f(xi) for xi in x])

    # Initialize the array to store Fourier coefficients
    if isinstance(degree, int):
        coefficients = np.zeros(2 * d + 1, dtype=complex)
    else:
        coefficients = np.zeros((2 * d1 + 1, 2 * d2 + 1), dtype=complex)

    # Compute the Fourier coefficients
    for idx, k in enumerate(k_values):
        if isinstance(degree, int):
            # For 1D case
            integrand = fx * np.exp(-1j * k * x)
            coefficients[idx] = np.sum(integrand) * dx / (2 * np.pi)
        else:
            # For 2D case
            k1, k2 = k
            integrand = fx * np.exp(-1j * (k1 * x[0] + k2 * x[1]))
            coefficients[idx // (2 * d2 + 1), idx % (2 * d2 + 1)] = np.sum(integrand) * dx / (2 * np.pi)

    return coefficients

