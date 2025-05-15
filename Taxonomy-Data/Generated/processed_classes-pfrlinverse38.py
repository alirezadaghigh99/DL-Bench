import torch
import torch.nn as nn
import numpy as np

class EmpiricalNormalization(nn.Module):
    def __init__(
        self,
        shape,
        batch_axis=0,
        eps=1e-2,
        dtype=np.float32,
        until=None,
        clip_threshold=None,
    ):
        super(EmpiricalNormalization, self).__init__()
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self.register_buffer(
            "_mean",
            torch.tensor(np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis))
        )
        self.register_buffer(
            "_var",
            torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis))
        )
        self.register_buffer("count", torch.tensor(0))

    def inverse(self, y):
        # Calculate the standard deviation
        std = torch.sqrt(self._var + self.eps)
        
        # Denormalize the input
        denormalized_y = y * std + self._mean
        
        return denormalized_y