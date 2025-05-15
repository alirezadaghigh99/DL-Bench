import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    # Validate inputs
    n = len(weight_floor)
    
    if len(weight_ceil) != n:
        raise ValueError("weight_floor and weight_ceil must have the same length.")
    
    if len(values) != 2**n:
        raise ValueError(f"values must have length 2^n, where n is the length of weight_floor and weight_ceil. Expected {2**n}, got {len(values)}.")
    
    # Ensure all tensors in values have the same shape
    value_shape = values[0].shape
    for v in values:
        if v.shape != value_shape:
            raise ValueError("All tensors in values must have the same shape.")
    
    # Perform linear interpolation
    result = torch.zeros_like(values[0])
    
    for i in range(2**n):
        # Calculate the weight for the i-th corner
        weight = torch.ones_like(values[0])
        for j in range(n):
            if (i >> j) & 1:
                weight *= weight_ceil[j]
            else:
                weight *= weight_floor[j]
        
        # Add the weighted value to the result
        result += weight * values[i]
    
    return result

