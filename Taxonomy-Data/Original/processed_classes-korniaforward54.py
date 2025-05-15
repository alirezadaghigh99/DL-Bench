    def forward(self, input: Tensor) -> Tensor:
        return vflip(input)