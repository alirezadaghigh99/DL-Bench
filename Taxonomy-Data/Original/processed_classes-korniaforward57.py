    def forward(self, input: Tensor) -> Tensor:
        return adjust_saturation(input, self.saturation_factor)