    def forward(self, input: Tensor) -> Tensor:
        return resize(
            input,
            self.size,
            self.interpolation,
            align_corners=self.align_corners,
            side=self.side,
            antialias=self.antialias,
        )