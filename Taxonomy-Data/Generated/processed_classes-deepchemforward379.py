    def forward(self, x: Tensor) -> Tensor:
        # Pass the input through the model
        out = x
        for layer in self.model:
            out = layer(out)
            if isinstance(layer, nn.Linear):  # Apply activation after each Linear layer
                out = self.activation_fn(out)

        # Handle skip connection if it exists
        if self.skip is not None:
            skip_out = self.skip(x)
            if self.weighted_skip:
                out = out + skip_out
            else:
                out = torch.cat((out, skip_out), dim=-1)

        return out