    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.LongTensor or torch.cuda.LongTensor
            input tensor (batch_size, max_sequence_length) with padded sequences of word ids
        """
        x = self._forward_pooled(x)
        return self._dropout_and_fc(x)