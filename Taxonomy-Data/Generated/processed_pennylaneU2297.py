import numpy as np

def U2(phi: float, delta: float) -> np.ndarray:
    # Calculate the complex exponential terms
    exp_i_delta = np.exp(1j * delta)
    exp_i_phi = np.exp(1j * phi)
    exp_i_phi_plus_delta = np.exp(1j * (phi + delta))
    
    # Construct the U2 matrix
    u2_matrix = (1 / np.sqrt(2)) * np.array([
        [1, -exp_i_delta],
        [exp_i_phi, exp_i_phi_plus_delta]
    ])
    
    return u2_matrix

