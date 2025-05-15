import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Fit:
    def __init__(self, model):
        self.model = model
        self.mean = None

    def fit(self, train_loader: DataLoader, override: bool = True, progress_bar: bool = False):
        if not override:
            raise ValueError("Override must be set to True to proceed with fitting.")

        # Set the model to evaluation mode
        self.model.eval()

        # Find the last layer of the model
        last_layer = None
        for layer in self.model.children():
            last_layer = layer

        if last_layer is None:
            raise ValueError("The model does not have any layers.")

        # Initialize parameters for fitting
        # Assuming last_layer is a linear layer for simplicity
        if isinstance(last_layer, nn.Linear):
            weight_mean = last_layer.weight.data.clone()
            bias_mean = last_layer.bias.data.clone() if last_layer.bias is not None else None
        else:
            raise NotImplementedError("Currently, only linear layers are supported.")

        # Fit the model using the train_loader
        for inputs, targets in train_loader:
            # Here you would implement the fitting logic, e.g., computing the Laplace approximation
            # This is a placeholder for the actual fitting logic
            pass

        # Set the mean parameter
        self.mean = weight_mean

        # Detach the mean if backpropagation is disabled
        if not torch.is_grad_enabled():
            self.mean = self.mean.detach()

        if progress_bar:
            print("Fitting completed.")

