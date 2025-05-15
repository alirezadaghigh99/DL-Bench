import torch

def gaussian_noise_image(image, mean=0.0, sigma=0.1, clip=True):
    """
    Adds Gaussian noise to an input image tensor.

    Parameters:
    - image (torch.Tensor): The input image tensor, expected to be of a floating-point dtype.
    - mean (float): Mean of the Gaussian noise. Default is 0.0.
    - sigma (float): Standard deviation of the Gaussian noise. Default is 0.1.
    - clip (bool): Whether to clip the output values to the [0, 1] range. Default is True.

    Returns:
    - torch.Tensor: The noisy image tensor.
    """
    if not image.is_floating_point():
        raise ValueError(f"Input tensor is expected to be in float dtype, got dtype={image.dtype}")
    
    if sigma < 0:
        raise ValueError(f"sigma shouldn't be negative. Got {sigma}")

    # Generate Gaussian noise
    noise = torch.randn_like(image) * sigma + mean
    
    # Add noise to the image
    noisy_image = image + noise
    
    # Clip the values to [0, 1] if required
    if clip:
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
    return noisy_image