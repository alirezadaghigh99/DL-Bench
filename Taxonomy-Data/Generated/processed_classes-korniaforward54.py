import torch
from torch import Tensor
from torch.nn import Module

class Rot180(Module):
    def forward(self, input: Tensor) -> Tensor:
        # Flip the tensor along the last two dimensions (H and W)
        return input.flip(-1).flip(-2)

