    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None,
                 actv=nn.Tanh, hidden_units=None):
        r"""Initializer method.
        """
        super(FCNN, self).__init__()

        # FORWARD COMPATIBILITY
        # If only one of {n_hidden_unit, n_hidden_layers} is specified, fill-in the other one
        if n_hidden_units is None and n_hidden_layers is not None:
            n_hidden_units = 32
        elif n_hidden_units is not None and n_hidden_layers is None:
            n_hidden_layers = 1

        # FORWARD COMPATIBILITY
        # When {n_hidden_unit, n_hidden_layers} are specified, construct an equivalent hidden_units if None is provided
        if n_hidden_units is not None or n_hidden_layers is not None:
            if hidden_units is None:
                hidden_units = tuple(n_hidden_units for _ in range(n_hidden_layers + 1))
                warn(f"`n_hidden_units` and `n_hidden_layers` are deprecated, "
                     f"pass `hidden_units={hidden_units}` instead",
                     FutureWarning)
            else:
                warn(f"Ignoring `n_hidden_units` and `n_hidden_layers` in favor of `hidden_units={hidden_units}`",
                     FutureWarning)

        # If none of {n_hidden_units, n_hidden_layers, hidden_layers} are specified, use (32, 32) by default
        if hidden_units is None:
            hidden_units = (32, 32)

        # If user passed in a list, iterator, etc., convert it to tuple
        if not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        units = (n_input_units,) + hidden_units
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(actv())
        # There's not activation in after the last layer
        layers.append(nn.Linear(units[-1], n_output_units))
        self.NN = torch.nn.Sequential(*layers)