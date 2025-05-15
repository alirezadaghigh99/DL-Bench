import torch
import torch.nn as nn

class MMCRLoss(nn.Module):
    def __init__(self, lmda: float = 5e-3):
        super().__init__()
        if lmda < 0:
            raise ValueError("lmda must be greater than or equal to 0")

        self.lmda = lmda

    def forward(self, online: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        # Ensure the online and momentum tensors have the same shape
        if online.shape != momentum.shape:
            raise ValueError("The 'online' and 'momentum' tensors must have the same shape.")

        # Concatenate online and momentum along the second dimension
        concatenated = torch.cat((online, momentum), dim=1)

        # Compute the centroid of the concatenated tensor
        centroid = torch.mean(concatenated, dim=0, keepdim=True)

        # Calculate the singular values of the concatenated tensor
        _, singular_values_concat, _ = torch.svd(concatenated)

        # Calculate the singular values of the centroid
        _, singular_values_centroid, _ = torch.svd(centroid)

        # Compute the loss
        batch_size = concatenated.size(0)
        loss = -torch.sum(singular_values_centroid) + self.lmda * torch.sum(singular_values_concat)
        loss = loss / batch_size

        return loss

