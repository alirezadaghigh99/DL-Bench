    def forward(
        self, X: Tensor, prior: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self._get_embeddings(X)
        steps_output, M_loss = self.encoder(x, prior)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return (res, M_loss)