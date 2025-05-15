import torch
import numpy as np
import torch.nn as nn

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
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

        # cache
        self._cached_std_inverse = None

    def experience(self, x):
        # Check if the batch size is zero
        if x.size(self.batch_axis) == 0:
            return

        # Compute the batch mean and variance
        batch_mean = x.mean(dim=self.batch_axis, keepdim=True)
        batch_var = x.var(dim=self.batch_axis, unbiased=False, keepdim=True)

        # Get the current count
        current_count = self.count.item()

        # Check if we should update
        if self.until is not None and current_count >= self.until:
            return

        # Update the count
        batch_size = x.size(self.batch_axis)
        new_count = current_count + batch_size
        self.count += batch_size

        # Compute the new mean and variance using a weighted average
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * (batch_size / new_count)
        m_a = self._var * current_count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta.pow(2) * current_count * batch_size / new_count
        new_var = M2 / new_count

        # Update the buffers
        self._mean.copy_(new_mean)
        self._var.copy_(new_var)
