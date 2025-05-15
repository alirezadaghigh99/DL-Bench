import torch
from typing import Optional

class Transform3d:
    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

def _handle_input(x, y, z, dtype, device, name):
    # This is a placeholder for the actual _handle_input function
    # Assuming it returns a tensor of shape (N, 3) for translation vectors
    if y is None and z is None:
        xyz = torch.tensor(x, dtype=dtype, device=device).view(-1, 3)
    else:
        xyz = torch.tensor([x, y, z], dtype=dtype, device=device).view(-1, 3)
    return xyz

class Translate(Transform3d):
    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        xyz = _handle_input(x, y, z, dtype, device, "Translate")
        super().__init__(device=xyz.device, dtype=dtype)
        N = xyz.shape[0]

        mat = torch.eye(4, dtype=dtype, device=self.device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 3, :3] = xyz
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        # Create an inverse mask
        inverse_matrix = self._matrix.clone()
        # Invert the translation part
        inverse_matrix[:, 3, :3] = -self._matrix[:, 3, :3]
        return inverse_matrix

