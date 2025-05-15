import numpy as np

def take_static_crop(image: np.ndarray, crop_params: dict) -> np.ndarray:
    """
    Crops an image based on the specified crop parameters.

    Parameters:
    - image: np.ndarray, the input image to be cropped.
    - crop_params: dict, a dictionary containing the crop parameters with keys:
        - 'x_min': float, minimum x percentage (0 to 1) of the image width.
        - 'y_min': float, minimum y percentage (0 to 1) of the image height.
        - 'x_max': float, maximum x percentage (0 to 1) of the image width.
        - 'y_max': float, maximum y percentage (0 to 1) of the image height.

    Returns:
    - np.ndarray, the cropped image.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate pixel values for the crop boundaries
    x_min = int(crop_params['x_min'] * width)
    y_min = int(crop_params['y_min'] * height)
    x_max = int(crop_params['x_max'] * width)
    y_max = int(crop_params['y_max'] * height)

    # Ensure the crop boundaries are within the image dimensions
    x_min = max(0, min(x_min, width))
    y_min = max(0, min(y_min, height))
    x_max = max(0, min(x_max, width))
    y_max = max(0, min(y_max, height))

    # Crop the image using the calculated boundaries
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image