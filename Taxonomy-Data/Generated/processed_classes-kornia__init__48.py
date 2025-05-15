import torch
from torch import nn
from typing import Union, Tuple, List

class Normalize(nn.Module):
    def __init__(
        self,
        mean: Union[torch.Tensor, Tuple[float], List[float], float],
        std: Union[torch.Tensor, Tuple[float], List[float], float],
    ) -> None:
        super().__init__()
        
        # Convert mean to a tensor if it is not already
        if not isinstance(mean, torch.Tensor):
            if isinstance(mean, (tuple, list)):
                self.mean = torch.tensor(mean, dtype=torch.float32)
            else:  # it's a float
                self.mean = torch.tensor([mean], dtype=torch.float32)
        else:
            self.mean = mean
        
        # Convert std to a tensor if it is not already
        if not isinstance(std, torch.Tensor):
            if isinstance(std, (tuple, list)):
                self.std = torch.tensor(std, dtype=torch.float32)
            else:  # it's a float
                self.std = torch.tensor([std], dtype=torch.float32)
        else:
            self.std = std