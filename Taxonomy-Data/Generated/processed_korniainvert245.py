import torch

def invert(image_tensor, max_value_tensor=None):
    # Check if the input image is a tensor
    assert isinstance(image_tensor, torch.Tensor), "Input image must be a tensor."
    
    # If max_value_tensor is not provided, assume it to be 1
    if max_value_tensor is None:
        max_value_tensor = torch.tensor(1.0, dtype=image_tensor.dtype, device=image_tensor.device)
    
    # Check if the max_value_tensor is a tensor
    assert isinstance(max_value_tensor, torch.Tensor), "Maximum value must be a tensor."
    
    # Invert the image tensor
    inverted_tensor = max_value_tensor - image_tensor
    
    return inverted_tensor

