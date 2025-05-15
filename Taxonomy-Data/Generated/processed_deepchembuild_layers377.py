import torch
import torch.nn as nn
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
        
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden if d_hidden is not None else []
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.activation_fn = activation_fn
        self.skip_connection = skip_connection
        self.weighted_skip = weighted_skip

        self.layers = self.build_layers()

    def build_layers(self):
        layer_list = []
        input_dim = self.d_input

        # Determine the activation function
        if isinstance(self.activation_fn, str):
            if self.activation_fn.lower() == 'relu':
                activation = nn.ReLU()
            elif self.activation_fn.lower() == 'tanh':
                activation = nn.Tanh()
            elif self.activation_fn.lower() == 'sigmoid':
                activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation function: {self.activation_fn}")
        else:
            activation = self.activation_fn

        # Iterate through hidden dimensions to create layers
        for hidden_dim in self.d_hidden:
            layer_list.append(nn.Linear(input_dim, hidden_dim))
            if self.batch_norm:
                layer_list.append(nn.BatchNorm1d(hidden_dim, momentum=self.batch_norm_momentum))
            layer_list.append(activation)
            if self.dropout > 0.0:
                layer_list.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim

        # Output layer
        layer_list.append(nn.Linear(input_dim, self.d_output))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)

