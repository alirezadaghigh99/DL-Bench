import torch
import torch.nn as nn
from torch.linalg import svd

class MMCRLoss(nn.Module):
    """Implementation of the loss function from MMCR [0] using Manifold Capacity.
    All hyperparameters are set to the default values from the paper for ImageNet.

    - [0]: Efficient Coding of Natural Images using Maximum Manifold Capacity
    Representations, 2023, https://arxiv.org/pdf/2303.03307.pdf

    Examples:
        >>> # initialize loss function
        >>> loss_fn = MMCRLoss()
        >>> transform = MMCRTransform(k=2)
        >>>
        >>> # transform images, then feed through encoder and projector
        >>> x = transform(x)
        >>> online = online_network(x)
        >>> momentum = momentum_network(x)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(online, momentum)
    """

    def __init__(self, lmda: float=0.005):
        """Initializes the MMCRLoss module with the specified lambda parameter.

        Args:
            lmda: The regularization parameter.

        Raises:
            ValueError: If lmda is less than 0.
        """
        super().__init__()
        if lmda < 0:
            raise ValueError('lmda must be greater than or equal to 0')
        self.lmda = lmda

    def forward(self, online: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """Computes the MMCR loss for the given online and momentum outputs.

        Args:
            online: Output of the online network.
            momentum: Output of the momentum network.

        Returns:
            The computed loss.
        """
        assert (
            online.shape == momentum.shape
        ), f"online and momentum need to have the same shape but are {online.shape} and {momentum.shape}"

        batch_size = online.shape[0]

        z = torch.cat([online, momentum], dim=1)
        centroid = torch.mean(z, dim=1)

        _, centroid_singular_values, _ = svd(centroid, full_matrices=False)
        _, z_singular_values, _ = svd(z, full_matrices=False)

        loss = (
            -torch.sum(centroid_singular_values)
            + self.lmda * torch.sum(z_singular_values)
        ) / batch_size

        return loss
