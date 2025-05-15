    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return dice_loss(pred, target, self.average, self.eps, self.weight, self.ignore_index)