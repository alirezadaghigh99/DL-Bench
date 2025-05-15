import torch
from torch import Tensor
from typing import Dict, Any

class RandomCrop3D:
    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        # Extract the crop parameters
        crop_size = params.get('crop_size')  # Expected to be a tensor of shape (3,)
        start_coords = params.get('start_coords')  # Expected to be a tensor of shape (3,)
        
        if crop_size is None or start_coords is None:
            raise ValueError("Parameters 'crop_size' and 'start_coords' must be provided.")
        
        # Create an identity transformation matrix
        transformation_matrix = torch.eye(4, dtype=input.dtype, device=input.device)
        
        # Set the translation part of the transformation matrix
        # The translation is negative of the start coordinates to simulate the crop
        transformation_matrix[0, 3] = -start_coords[0].item()
        transformation_matrix[1, 3] = -start_coords[1].item()
        transformation_matrix[2, 3] = -start_coords[2].item()
        
        return transformation_matrix

