import torch
from typing import Any, Dict, Tuple, Type, Union, Optional
import warnings
from torchvision import tv_tensors

# Assuming Transform is a base class that you have defined elsewhere
class Transform:
    pass

def is_pure_tensor(obj):
    return isinstance(obj, torch.Tensor)

def _get_defaultdict(dims):
    # This function should return a defaultdict-like object that provides default dims
    # For simplicity, let's assume it returns a dictionary with default values
    return {torch.Tensor: dims, tv_tensors.Image: dims, tv_tensors.Video: dims}

class TransposeDimensions(Transform):
    _transformed_types = (is_pure_tensor, tv_tensors.Image, tv_tensors.Video)

    def __init__(self, dims: Union[Tuple[int, int], Dict[Type, Optional[Tuple[int, int]]]]) -> None:
        super().__init__()
        if not isinstance(dims, dict):
            dims = _get_defaultdict(dims)
        if torch.Tensor in dims and any(cls in dims for cls in [tv_tensors.Image, tv_tensors.Video]):
            warnings.warn(
                "Got `dims` values for `torch.Tensor` and either `tv_tensors.Image` or `tv_tensors.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `tv_tensors.Image` or `tv_tensors.Video` is present in the input."
            )
        self.dims = dims

    def _transform(self, input: Any, params: Dict[str, Any]) -> torch.Tensor:
        # Determine the type of the input
        input_type = type(input)

        # Check if the input type is one of the transformed types
        if any(isinstance(input, t) for t in self._transformed_types):
            # Get the dimensions to transpose for this type
            dims = self.dims.get(input_type)

            # If dimensions are specified, transpose the input
            if dims is not None:
                return input.transpose(*dims)

        # If no specific dimensions are provided, return the input as a torch.Tensor
        return torch.as_tensor(input)

