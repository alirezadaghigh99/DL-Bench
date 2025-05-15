import numpy as np

def draw_point2d(image, points, color):
    """
    Draws points on an image tensor with the specified color.

    Parameters:
    - image: numpy.ndarray, the image tensor, either (H, W) for grayscale or (C, H, W) for multi-channel.
    - points: list of tuples, each tuple is (x, y) representing the coordinates to be colored.
    - color: numpy.ndarray or scalar, the color to set at the specified points. Should match the image's channels.

    Returns:
    - numpy.ndarray, the modified image with points colored.
    """
    # Check if the image is grayscale or multi-channel
    if image.ndim == 2:
        # Grayscale image (H, W)
        H, W = image.shape
        for x, y in points:
            if 0 <= x < W and 0 <= y < H:
                image[y, x] = color
    elif image.ndim == 3:
        # Multi-channel image (C, H, W)
        C, H, W = image.shape
        for x, y in points:
            if 0 <= x < W and 0 <= y < H:
                image[:, y, x] = color
    else:
        raise ValueError("Image must be either 2D (H, W) or 3D (C, H, W).")

    return image

