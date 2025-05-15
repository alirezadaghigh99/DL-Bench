import timm
import torch

def resnet50(weights_path=None, *args, **kwargs):
    """
    Creates a ResNet-50 model using the timm library.

    Parameters:
    - weights_path (str, optional): Path to the pre-trained model weights.
    - *args: Additional arguments for the timm.create_model function.
    - **kwargs: Additional keyword arguments for the timm.create_model function.

    Returns:
    - model (torch.nn.Module): The ResNet-50 model.
    """
    # Determine the number of input channels based on the weights
    if weights_path:
        # Load the state dictionary from the provided weights
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Check for expected keys in the state dictionary
        expected_keys = ['conv1.weight', 'fc.weight', 'fc.bias']
        for key in expected_keys:
            if key not in state_dict:
                raise ValueError(f"Expected key '{key}' not found in the state dictionary.")
        
        # Determine input channels from the weights
        input_channels = state_dict['conv1.weight'].shape[1]
    else:
        # Default input channels for ResNet-50
        input_channels = 3

    # Create the ResNet-50 model
    model = timm.create_model('resnet50', in_chans=input_channels, *args, **kwargs)

    # Load the weights if provided
    if weights_path:
        model.load_state_dict(state_dict)

    return model