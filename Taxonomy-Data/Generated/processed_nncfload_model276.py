import torch
import torchvision.models as models
import os
import urllib
from torch.hub import load_state_dict_from_url

# Assuming custom_models is a module with custom model definitions
# import custom_models

def load_model(model_name, pretrained=False, num_classes=1000, model_params=None, weights_path=None):
    """
    Load a machine learning model using PyTorch.

    Parameters:
    - model_name (str): The name of the model to load.
    - pretrained (bool): Whether to load pretrained weights.
    - num_classes (int): The number of classes for the model.
    - model_params (dict): Additional parameters for the model.
    - weights_path (str): Path to custom weights.

    Returns:
    - model (torch.nn.Module): The loaded model.
    """
    model_params = model_params or {}

    # Load model from torchvision.models
    if hasattr(models, model_name):
        model_class = getattr(models, model_name)
        model = model_class(pretrained=pretrained, **model_params)
    # Load model from custom_models
    # elif hasattr(custom_models, model_name):
    #     model_class = getattr(custom_models, model_name)
    #     model = model_class(**model_params)
    else:
        raise ValueError(f"Model {model_name} is not defined in torchvision.models or custom_models.")

    # Modify the final layer to match the number of classes
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Linear):
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model.classifier, torch.nn.Sequential):
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError("The model architecture is not supported for modifying the final layer.")

    # Load custom weights if provided
    if weights_path:
        if not pretrained:
            if os.path.isfile(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
            elif urllib.parse.urlparse(weights_path).scheme in ('http', 'https'):
                state_dict = load_state_dict_from_url(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                raise ValueError(f"Invalid weights path: {weights_path}")

    return model

