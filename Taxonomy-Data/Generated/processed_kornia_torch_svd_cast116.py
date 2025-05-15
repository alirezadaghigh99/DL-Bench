import torch

def _torch_svd_cast(tensor):
    # Check if the input is a tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    
    # Check the data type of the tensor
    original_dtype = tensor.dtype
    if original_dtype not in [torch.float32, torch.float64]:
        # Cast the tensor to float32 if it's not float32 or float64
        tensor = tensor.to(torch.float32)
    
    # Perform SVD
    U, S, V = torch.svd(tensor)
    
    # Return the components of the SVD
    return U, S, V

