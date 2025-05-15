import torch
from torch import Tensor
from typing import Optional, Dict, Any

class GeometricAugmentationBase3D:
    def precrop_padding(self, input: Tensor, flags: Optional[Dict[str, Any]] = None) -> Tensor:
        # Default padding values
        padding = (0, 0, 0, 0, 0, 0)  # No padding by default (z1, z2, y1, y2, x1, x2)

        if flags is not None:
            # Extract padding values from flags if provided
            z_pad = flags.get('z_pad', 0)
            y_pad = flags.get('y_pad', 0)
            x_pad = flags.get('x_pad', 0)
            # Create padding tuple for 3D tensor (z1, z2, y1, y2, x1, x2)
            padding = (x_pad, x_pad, y_pad, y_pad, z_pad, z_pad)

        # Apply padding to the input tensor
        padded_tensor = torch.nn.functional.pad(input, padding, mode='constant', value=0)

        return padded_tensor

