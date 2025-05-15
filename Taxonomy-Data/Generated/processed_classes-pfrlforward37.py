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

        # cache
        self._cached_std_inverse = None

    def experience(self, x):
        """Update running mean and variance based on input `x`.

        Args:
            x (torch.Tensor): Input values of shape `(batch_size, ...)`.
                The method updates the internal `_mean` and `_var` buffers
                to reflect the mean and variance of the data observed so far.

        """
        batch_size = x.size(self.batch_axis)
        batch_mean = x.mean(dim=self.batch_axis, keepdim=True)
        batch_var = x.var(dim=self.batch_axis, unbiased=False, keepdim=True)

        # Update count
        new_count = self.count + batch_size

        # Update mean
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * batch_size / new_count

        # Update variance
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_size / new_count
        new_var = M2 / new_count

        # Update buffers
        self._mean = new_mean
        self._var = new_var
        self.count = new_count

    def forward(self, x, update=True):
        """Normalize input `x` using the running mean and variance.

        Args:
            x (torch.Tensor): Input values of shape `(batch_size, ...)`.
            update (bool): If True, update the running statistics.

        Returns:
            torch.Tensor: Normalized input.
        """
        if update:
            self.experience(x)

        # Calculate standard deviation with epsilon for numerical stability
        std = torch.sqrt(self._var + self.eps)

        # Normalize
        x_normalized = (x - self._mean) / std

        # Optionally clip the normalized values
        if self.clip_threshold is not None:
            x_normalized = torch.clamp(x_normalized, -self.clip_threshold, self.clip_threshold)

        return x_normalized