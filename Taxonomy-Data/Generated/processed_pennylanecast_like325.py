import numpy as np

def cast_like(tensor1, tensor2):
    # Convert tensor1 and tensor2 to numpy arrays if they are lists or tuples
    if isinstance(tensor1, (list, tuple)):
        tensor1 = np.array(tensor1)
    if isinstance(tensor2, (list, tuple)):
        tensor2 = np.array(tensor2)
    
    # Get the data type of tensor2
    target_dtype = tensor2.dtype
    
    # Cast tensor1 to the data type of tensor2
    casted_tensor1 = tensor1.astype(target_dtype)
    
    # If the original tensor1 was a list or tuple, convert it back to that type
    if isinstance(tensor1, np.ndarray):
        if isinstance(tensor1.tolist(), list):
            return casted_tensor1.tolist()
        elif isinstance(tensor1.tolist(), tuple):
            return tuple(casted_tensor1.tolist())
    
    return casted_tensor1

