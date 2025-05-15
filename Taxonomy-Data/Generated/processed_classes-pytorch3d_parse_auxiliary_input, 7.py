import torch
from typing import List, Optional, Tuple, Union

class Pointclouds:
    _INTERNAL_TENSORS = [
        "_points_packed",
        "_points_padded",
        "_normals_packed",
        "_normals_padded",
        "_features_packed",
        "_features_padded",
        "_packed_to_cloud_idx",
        "_cloud_to_packed_first_idx",
        "_num_points_per_cloud",
        "_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    def __init__(self, points, normals=None, features=None) -> None:
        self.device = torch.device("cpu")
        self.equisized = False
        self.valid = None
        self._N = 0
        self._P = 0
        self._C = None
        self._points_list = None
        self._normals_list = None
        self._features_list = None
        self._num_points_per_cloud = None
        self._points_packed = None
        self._normals_packed = None
        self._features_packed = None
        self._packed_to_cloud_idx = None
        self._cloud_to_packed_first_idx = None
        self._points_padded = None
        self._normals_padded = None
        self._features_padded = None
        self._padded_to_packed_idx = None
        # initialization code...

    def _parse_auxiliary_input(
        self, aux_input: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[int]]:
        if aux_input is None:
            return None, None, None

        if isinstance(aux_input, list):
            # Check if the list is empty
            if len(aux_input) == 0:
                return [], None, 0

            # Ensure all tensors in the list have the same number of channels
            num_channels = aux_input[0].shape[1]
            for tensor in aux_input:
                if tensor.shape[1] != num_channels:
                    raise ValueError("All tensors in the list must have the same number of channels.")

            # Return the list, None for the padded tensor, and the number of channels
            return aux_input, None, num_channels

        elif isinstance(aux_input, torch.Tensor):
            # Check if the tensor is 3D
            if aux_input.dim() != 3:
                raise ValueError("Padded tensor must be 3D with shape (num_clouds, num_points, C).")

            num_channels = aux_input.shape[2]
            # Convert the padded tensor to a list of tensors
            aux_list = [aux_input[i, :].squeeze(0) for i in range(aux_input.shape[0])]

            # Return the list, the padded tensor, and the number of channels
            return aux_list, aux_input, num_channels

        else:
            raise TypeError("aux_input must be a list of tensors or a padded tensor.")