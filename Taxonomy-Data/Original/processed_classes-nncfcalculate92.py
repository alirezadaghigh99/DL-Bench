    def calculate(self) -> torch.Tensor:
        if self.disabled:
            return 0

        params = 0
        loss = 0
        sparse_prob_sum = 0
        for sparse_layer in self._sparse_layers:
            if not self.disabled and sparse_layer.frozen:
                raise AssertionError(
                    "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss"
                )
            if not sparse_layer.frozen:
                sw_loss = sparse_layer.loss()
                params = params + sw_loss.view(-1).size(0)
                loss = loss + sw_loss.sum()
                sparse_prob_sum += torch.sigmoid(sparse_layer.mask).sum()

        self.mean_sparse_prob = (sparse_prob_sum / params).item()
        self.current_sparsity = 1 - loss / params
        return ((loss / params - self.target) / self.p).pow(2)