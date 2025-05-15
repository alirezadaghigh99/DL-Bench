import torch
import torch.nn as nn
from warnings import warn

class FCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None,
                 actv=nn.Tanh, hidden_units=None):
        super(FCNN, self).__init__()

        # Handle deprecated parameters
        if n_hidden_units is not None or n_hidden_layers is not None:
            warn("Parameters 'n_hidden_units' and 'n_hidden_layers' are deprecated. "
                 "Please use 'hidden_units' instead.", DeprecationWarning)
            if hidden_units is None:
                if n_hidden_units is not None and n_hidden_layers is not None:
                    hidden_units = (n_hidden_units,) * n_hidden_layers
                else:
                    hidden_units = (32, 32)
        
        # Default hidden_units if not provided
        if hidden_units is None:
            hidden_units = (32, 32)

        # Construct the network
        layers = []
        input_size = n_input_units

        for hidden_size in hidden_units:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(actv())
            input_size = hidden_size

        # Add the final output layer without activation
        layers.append(nn.Linear(input_size, n_output_units))

        # Store the network as a sequential model
        self.NN = nn.Sequential(*layers)

    def forward(self, x):
        return self.NN(x)