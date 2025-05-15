import torch
import torch.nn.functional as F

def _resize_image_and_masks(image, self_min_size, self_max_size, target=None, fixed_size=None):
    """
    Resizes an image tensor and its corresponding masks, if provided.

    Parameters:
    - image: A Tensor representing the image to be resized, with shape (C, H, W).
    - self_min_size: An integer specifying the minimum size for the image's smaller dimension when resizing.
    - self_max_size: An integer specifying the maximum size for the image's larger dimension when resizing.
    - target: An optional dictionary containing additional data, such as masks, that should be resized alongside the image.
    - fixed_size: An optional tuple (height, width) specifying the fixed dimensions to which the image should be resized.

    Returns:
    - A tuple containing:
      - The resized image tensor.
      - The resized target dictionary if it was provided, with resized masks if present.
    """
    # Determine the original size of the image
    _, original_height, original_width = image.shape

    if fixed_size is not None:
        # Resize to the fixed size
        new_height, new_width = fixed_size
    else:
        # Calculate the scaling factor
        min_original_size = float(min(original_height, original_width))
        max_original_size = float(max(original_height, original_width))

        # Compute the scaling factor based on the minimum size
        scale_factor = self_min_size / min_original_size

        # Ensure the maximum size constraint is not violated
        if max_original_size * scale_factor > self_max_size:
            scale_factor = self_max_size / max_original_size

        # Calculate the new dimensions
        new_height = int(round(original_height * scale_factor))
        new_width = int(round(original_width * scale_factor))

    # Resize the image
    resized_image = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

    # Resize the masks if they are present in the target
    if target is not None and "masks" in target:
        masks = target["masks"]
        # Ensure masks are in the correct shape (N, H, W)
        if masks.dim() == 3:
            resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(new_height, new_width), mode='nearest').squeeze(1)
            target["masks"] = resized_masks

    return resized_image, target

