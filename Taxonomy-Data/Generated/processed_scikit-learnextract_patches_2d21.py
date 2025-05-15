import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d as skl_extract_patches_2d

def extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    """
    Extracts patches from a 2D image.

    Parameters:
    - image: ndarray of shape (image_height, image_width, n_channels)
      The original image data.
    - patch_size: tuple of int (patch_height, patch_width)
      The dimensions of one patch.
    - max_patches: int, default=None
      The maximum number of patches to extract. If None, all patches are extracted.
    - random_state: int or RandomState, default=None
      Determines the random number generator for random sampling.

    Returns:
    - patches: ndarray of shape (n_patches, patch_height, patch_width, n_channels)
      The collection of patches extracted from the image.
    """
    # Use sklearn's extract_patches_2d to extract patches
    patches = skl_extract_patches_2d(image, patch_size, max_patches=max_patches, random_state=random_state)
    return patches

