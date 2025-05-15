import torch

def shift_rgb(image, r_shift, g_shift, b_shift):
    # Check if the input is a valid tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("The image must be a torch.Tensor.")
    
    # Check if the image has the correct shape
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("The image must have shape (N, 3, H, W) for an RGB image.")
    
    # Check if the shift tensors are valid
    if not (isinstance(r_shift, torch.Tensor) and isinstance(g_shift, torch.Tensor) and isinstance(b_shift, torch.Tensor)):
        raise TypeError("The shift values must be torch.Tensors.")
    
    if r_shift.ndim != 1 or g_shift.ndim != 1 or b_shift.ndim != 1:
        raise ValueError("The shift tensors must be 1-dimensional.")
    
    if r_shift.shape[0] != image.shape[0] or g_shift.shape[0] != image.shape[0] or b_shift.shape[0] != image.shape[0]:
        raise ValueError("The shift tensors must have the same length as the batch size of the image.")
    
    # Apply the shifts to each channel
    shifted_image = image.clone()  # Clone the image to avoid modifying the original
    shifted_image[:, 0, :, :] += r_shift.view(-1, 1, 1)  # Red channel
    shifted_image[:, 1, :, :] += g_shift.view(-1, 1, 1)  # Green channel
    shifted_image[:, 2, :, :] += b_shift.view(-1, 1, 1)  # Blue channel
    
    # Clamp the values to be between 0 and 1
    shifted_image = torch.clamp(shifted_image, 0, 1)
    
    return shifted_image

