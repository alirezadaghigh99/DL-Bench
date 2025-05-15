import torch

def total_variation(image_tensor, reduction='sum'):
    """
    Computes the Total Variation of an input image tensor.

    Parameters:
    - image_tensor (torch.Tensor): The input image tensor with shape (*, H, W).
    - reduction (str): Specifies the reduction method, either 'sum' or 'mean'.

    Returns:
    - torch.Tensor: A tensor with shape (*) representing the total variation.
    """
    # Check if the input is a torch.Tensor
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Check if the reduction parameter is valid
    if reduction not in ['sum', 'mean']:
        raise ValueError("Reduction must be either 'sum' or 'mean'")

    # Calculate the differences along the height dimension
    diff_h = torch.abs(image_tensor[..., 1:, :] - image_tensor[..., :-1, :])

    # Calculate the differences along the width dimension
    diff_w = torch.abs(image_tensor[..., :, 1:] - image_tensor[..., :, :-1])

    # Sum the differences
    total_variation_value = diff_h.sum(dim=(-2, -1)) + diff_w.sum(dim=(-2, -1))

    # Apply the reduction method
    if reduction == 'mean':
        num_elements = image_tensor.shape[-2] * image_tensor.shape[-1]
        total_variation_value = total_variation_value / num_elements

    return total_variation_value

