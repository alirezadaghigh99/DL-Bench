import numpy as np

def normalized_mse(ref_outputs, approx_outputs):
    """
    Compute the normalized mean square error (NMSE) between two lists of NumPy arrays.

    Parameters:
    - ref_outputs: List of NumPy arrays representing the reference outputs.
    - approx_outputs: List of NumPy arrays representing the approximate outputs.

    Returns:
    - A float representing the average NMSE across all pairs of arrays.
    """
    if len(ref_outputs) != len(approx_outputs):
        raise ValueError("The lists ref_outputs and approx_outputs must have the same length.")
    
    nmse_values = []
    
    for ref, approx in zip(ref_outputs, approx_outputs):
        if ref.shape != approx.shape:
            raise ValueError("Each pair of arrays must have the same shape.")
        
        mse = np.mean((ref - approx) ** 2)
        mse_ref_zero = np.mean(ref ** 2)
        
        if mse_ref_zero == 0:
            raise ValueError("MSE between reference output and zero is zero, cannot normalize.")
        
        nmse = mse / mse_ref_zero
        nmse_values.append(nmse)
    
    average_nmse = np.mean(nmse_values)
    return average_nmse