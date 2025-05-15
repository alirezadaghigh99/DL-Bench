    def build_layers(self):
        """
        Build the layers of the model, iterating through the hidden dimensions to produce a list of layers.
        """

        layer_list = []
        layer_dim = self.d_input
        if self.d_hidden is not None:
            for d in self.d_hidden:
                layer_list.append(nn.Linear(layer_dim, d))
                layer_list.append(self.dropout)
                if self.batch_norm:
                    layer_list.append(
                        nn.BatchNorm1d(d, momentum=self.batch_norm_momentum))
                layer_dim = d
        layer_list.append(nn.Linear(layer_dim, self.d_output))
        return layer_list