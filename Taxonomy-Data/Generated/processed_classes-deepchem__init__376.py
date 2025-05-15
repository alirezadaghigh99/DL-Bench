import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable

class MultilayerPerceptron(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_hidden: Optional[tuple] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1,
                 activation_fn: Union[Callable, str] = 'relu',
                 skip_connection: bool = False,
                 weighted_skip: bool = True):
        super(MultilayerPerceptron, self).__init__()

        # Set activation function
        if isinstance(activation_fn, str):
            if activation_fn == 'relu':
                self.activation_fn = F.relu
            elif activation_fn == 'tanh':
                self.activation_fn = torch.tanh
            elif activation_fn == 'sigmoid':
                self.activation_fn = torch.sigmoid
            else:
                raise ValueError(f"Unsupported activation function: {activation_fn}")
        else:
            self.activation_fn = activation_fn

        # Define layers
        layers = []
        input_dim = d_input
        if d_hidden is not None:
            for hidden_dim in d_hidden:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU() if activation_fn == 'relu' else self.activation_fn)
                input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, d_output))

        self.layers = nn.Sequential(*layers)

        # Skip connection
        self.skip_connection = skip_connection
        self.weighted_skip = weighted_skip
        if skip_connection and weighted_skip:
            self.skip_weight = nn.Parameter(torch.ones(d_output))

    def forward(self, x):
        out = self.layers(x)
        if self.skip_connection:
            if self.weighted_skip:
                skip_out = x @ self.skip_weight
            else:
                skip_out = x
            out += skip_out
        return out