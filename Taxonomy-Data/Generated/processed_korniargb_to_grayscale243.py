import torch

def rgb_to_grayscale(image, rgb_weights=None):
    # Validate input shape
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("Input image must have shape (*, 3, H, W)")

    # Determine the data type and set default weights if not provided
    if rgb_weights is None:
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], dtype=torch.float32) / 255.0
        elif torch.is_floating_point(image):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype)
        else:
            raise TypeError("Unsupported image data type. Supported types are uint8 and floating-point.")

    # Validate that rgb_weights sum to 1
    if not torch.isclose(rgb_weights.sum(), torch.tensor(1.0, dtype=rgb_weights.dtype)):
        raise ValueError("The sum of rgb_weights must be 1.")

    # Ensure rgb_weights is a tensor
    rgb_weights = torch.tensor(rgb_weights, dtype=image.dtype)

    # Reshape rgb_weights to be broadcastable with the image
    rgb_weights = rgb_weights.view(1, 3, 1, 1)

    # Convert to grayscale
    grayscale_image = (image * rgb_weights).sum(dim=-3, keepdim=True)

    return grayscale_image

