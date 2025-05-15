import torch
import torch.nn.functional as F

def apply_blurpool_on_weights(weights):
    # Ensure the input is a tensor
    if not isinstance(weights, torch.Tensor):
        raise ValueError("Input weights must be a torch.Tensor")

    # Get the original shape
    original_shape = weights.shape

    # Pad the weights tensor along the last dimension
    # We use 'reflect' padding to handle edge cases
    padded_weights = F.pad(weights, (1, 1), mode='reflect')

    # Apply 2-tap max pooling
    # We use a kernel size of 2 and stride of 1 to get the max filter effect
    max_pooled_weights = F.max_pool1d(padded_weights.unsqueeze(1), kernel_size=2, stride=1).squeeze(1)

    # Apply 2-tap blur filter using linear interpolation
    # We use a simple average of adjacent elements to achieve the blur effect
    # This is equivalent to a 2-tap blur filter
    left_shifted = F.pad(max_pooled_weights, (0, 1), mode='reflect')[..., :-1]
    right_shifted = F.pad(max_pooled_weights, (1, 0), mode='reflect')[..., 1:]
    blurred_weights = (left_shifted + right_shifted) / 2

    # Ensure the output shape matches the input shape
    blurred_weights = blurred_weights[..., :original_shape[-1]]

    return blurred_weights

