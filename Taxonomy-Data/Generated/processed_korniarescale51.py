import torch
import torch.nn.functional as F

def rescale(input, factor, interpolation="bilinear", align_corners=None, antialias=False):
    # Ensure the input is a 4D tensor (batch_size, channels, height, width)
    if input.dim() != 4:
        raise ValueError("Input tensor must be 4-dimensional (batch_size, channels, height, width)")

    # Determine the scaling factors for height and width
    if isinstance(factor, (float, int)):
        factor = (factor, factor)
    elif isinstance(factor, tuple) and len(factor) == 2:
        pass
    else:
        raise ValueError("Factor must be a float, int, or a tuple of two floats/ints")

    # Calculate the new size
    _, _, original_height, original_width = input.shape
    new_height = int(original_height * factor[0])
    new_width = int(original_width * factor[1])

    # Use torch.nn.functional.interpolate to resize the tensor
    output = F.interpolate(
        input,
        size=(new_height, new_width),
        mode=interpolation,
        align_corners=align_corners,
        antialias=antialias
    )

    return output

