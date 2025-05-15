import torch

def rgb_to_rgba(image: torch.Tensor, alpha_val) -> torch.Tensor:
    """
    Convert an RGB image to RGBA by adding an alpha channel.

    Args:
        image (torch.Tensor): RGB Image to be converted to RGBA of shape (*, 3, H, W).
        alpha_val (float or torch.Tensor): A float number for the alpha value or a tensor
                                           of shape (*, 1, H, W).

    Returns:
        torch.Tensor: RGBA version of the image with shape (*, 4, H, W).
    """
    if image.shape[-3] != 3:
        raise ValueError("Input image must have 3 channels in the last dimension for RGB.")

    # If alpha_val is a float, create a tensor with the same shape as the image
    if isinstance(alpha_val, float):
        alpha_channel = torch.full_like(image[:, :1, :, :], alpha_val)
    elif isinstance(alpha_val, torch.Tensor):
        if alpha_val.shape[-3] != 1:
            raise ValueError("Alpha tensor must have 1 channel in the last dimension.")
        if alpha_val.shape != image.shape[:-3] + (1, image.shape[-2], image.shape[-1]):
            raise ValueError("Alpha tensor must have the same shape as the image except for the channel dimension.")
        alpha_channel = alpha_val
    else:
        raise TypeError("alpha_val must be either a float or a torch.Tensor.")

    # Concatenate the alpha channel to the RGB image
    rgba_image = torch.cat((image, alpha_channel), dim=-3)

    return rgba_image

