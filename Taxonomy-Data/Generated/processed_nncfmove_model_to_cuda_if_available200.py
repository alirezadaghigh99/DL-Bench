import torch

def move_model_to_cuda_if_available(model):
    """
    Moves the model to a CUDA device if available and returns the device of the first parameter.

    Parameters:
    model (torch.nn.Module): The PyTorch model to be moved.

    Returns:
    torch.device: The device of the first parameter of the model.
    """
    # Check if CUDA is available and move the model to CUDA if it is
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Get the device of the first parameter of the model
    first_param_device = next(model.parameters()).device
    
    return first_param_device

