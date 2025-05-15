import torch
from typing import Optional, Union
from torch import Tensor

class Translate(Transform3d):
    def __init__(
        self,
        x: Union[Tensor, float],
        y: Optional[Union[Tensor, float]] = None,
        z: Optional[Union[Tensor, float]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if isinstance(x, Tensor) and x.ndim == 2 and x.shape[1] == 3:
            # Case 1: x is a tensor of shape (N, 3)
            self.translations = x.to(dtype=dtype, device=device)
        elif y is not None and z is not None:
            # Case 2: x, y, z are provided as scalars or 1D tensors
            x = torch.tensor(x, dtype=dtype, device=device) if not isinstance(x, Tensor) else x.to(dtype=dtype, device=device)
            y = torch.tensor(y, dtype=dtype, device=device) if not isinstance(y, Tensor) else y.to(dtype=dtype, device=device)
            z = torch.tensor(z, dtype=dtype, device=device) if not isinstance(z, Tensor) else z.to(dtype=dtype, device=device)
            
            # Ensure x, y, z are 1D tensors
            if x.ndim == 0:
                x = x.unsqueeze(0)
            if y.ndim == 0:
                y = y.unsqueeze(0)
            if z.ndim == 0:
                z = z.unsqueeze(0)
            
            # Stack to form a (N, 3) tensor
            self.translations = torch.stack((x, y, z), dim=-1)
        else:
            raise ValueError("Invalid input: Provide either a tensor of shape (N, 3) or individual x, y, z values.")

        # Create the translation matrix
        N = self.translations.shape[0]
        self.matrix = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        self.matrix[:, :3, 3] = self.translations