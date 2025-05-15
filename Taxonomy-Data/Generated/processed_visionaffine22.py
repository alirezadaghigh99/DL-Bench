from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

def affine(img, angle, translate, scale, shear, interpolation=Image.BILINEAR, fill=0, center=None):
    """
    Apply an affine transformation to an image while keeping the image center invariant.

    Parameters:
    - img: PIL Image or Tensor
    - angle: Rotation angle in degrees
    - translate: Tuple of horizontal and vertical translations
    - scale: Overall scale factor
    - shear: Tuple of shear angles (shear_x, shear_y)
    - interpolation: Interpolation mode (e.g., Image.NEAREST, Image.BILINEAR, Image.BICUBIC)
    - fill: Pixel fill value for areas outside the transformed image
    - center: Optional center of rotation (tuple of x, y coordinates)

    Returns:
    - Transformed image as a PIL Image or Tensor
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    if center is None:
        # Default center is the center of the image
        center = (img.width / 2, img.height / 2)

    # Convert shear to a tuple if it's not already
    if isinstance(shear, (int, float)):
        shear = (shear, 0)

    # Apply the affine transformation
    transformed_img = TF.affine(
        img,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        fill=fill,
        center=center
    )

    return transformed_img

