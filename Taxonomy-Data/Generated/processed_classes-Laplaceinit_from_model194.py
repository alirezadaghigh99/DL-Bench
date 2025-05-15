import torch
import torch.nn as nn

class Kron:
    def __init__(self, kronecker_factors):
        self.kronecker_factors = kronecker_factors

    @classmethod
    def init_from_model(cls, model, device):
        if isinstance(model, nn.Module):
            parameters = model.parameters()
        elif isinstance(model, (list, tuple)) and all(isinstance(p, nn.Parameter) for p in model):
            parameters = model
        else:
            raise ValueError("Model must be an instance of nn.Module or an iterable of nn.Parameter.")

        kronecker_factors = []

        for param in parameters:
            param_shape = param.shape
            if len(param_shape) == 1:
                # Bias term: create a square matrix of zeros
                size = param_shape[0]
                kronecker_factors.append((torch.zeros((size, size), device=device),))
            elif len(param_shape) >= 2:
                # Fully connected or convolutional layers
                input_dim = param_shape[1]
                output_dim = param_shape[0]
                kronecker_factors.append((
                    torch.zeros((input_dim, input_dim), device=device),
                    torch.zeros((output_dim, output_dim), device=device)
                ))
            else:
                raise ValueError(f"Invalid parameter shape: {param_shape}")

        return cls(kronecker_factors)

