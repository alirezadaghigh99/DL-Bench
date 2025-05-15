import torch

def _assert_image_tensor(img):
    """Check if the input is a valid image tensor."""
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    if img.ndim < 3:
        raise TypeError("Image tensor must have at least 3 dimensions (C, H, W).")

def _assert_channels(img):
    """Check if the image tensor has either 1 or 3 channels."""
    if img.shape[0] not in (1, 3):
        raise TypeError("Image tensor must have 1 or 3 channels.")

def invert(img):
    """Invert the image tensor."""
    max_val = torch.iinfo(img.dtype).max
    return max_val - img

def solarize(img, threshold):
    """Apply solarization effect to the image tensor."""
    _assert_image_tensor(img)
    _assert_channels(img)

    # Check if the threshold is valid
    max_val = torch.iinfo(img.dtype).max
    if threshold > max_val:
        raise TypeError("Threshold value cannot be greater than the maximum value of the image tensor's data type.")

    # Invert the image
    inverted_img = invert(img)

    # Apply the solarization effect
    solarized_img = torch.where(img >= threshold, inverted_img, img)

    return solarized_img

