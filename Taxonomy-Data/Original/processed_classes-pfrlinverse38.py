    def inverse(self, y):
        std = torch.sqrt(self._var + self.eps)
        return y * std + self._mean