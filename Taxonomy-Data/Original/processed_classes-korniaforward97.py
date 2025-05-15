    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return lovasz_softmax_loss(pred=pred, target=target, weight=self.weight)