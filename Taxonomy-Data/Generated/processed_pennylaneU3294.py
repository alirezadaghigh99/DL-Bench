import numpy as np

def U3(theta, phi, delta):
    # Calculate the elements of the unitary matrix
    a = np.cos(theta / 2)
    b = -np.exp(1j * delta) * np.sin(theta / 2)
    c = np.exp(1j * phi) * np.sin(theta / 2)
    d = np.exp(1j * (phi + delta)) * np.cos(theta / 2)
    
    # Create the 2x2 unitary matrix
    unitary_matrix = np.array([[a, b],
                               [c, d]], dtype=complex)
    
    return unitary_matrix

