from PIL import Image, ExifTags
import torch
import torchvision.transforms as transforms

class ImageReadMode:
    RGB = 'RGB'
    GRAYSCALE = 'L'

def read_image(path, mode=ImageReadMode.RGB, apply_exif_orientation=True):
    """
    Reads a JPEG, PNG, or GIF image from a specified path and converts it into a 3-dimensional
    RGB or grayscale Tensor. The function can handle EXIF orientation if specified.

    Parameters:
    - path (str): The file path to the image.
    - mode (ImageReadMode, optional): The mode to read the image in. Can be 'RGB' or 'L' (grayscale).
      Defaults to 'RGB'.
    - apply_exif_orientation (bool, optional): Whether to apply EXIF orientation corrections.
      Defaults to True.

    Returns:
    - torch.Tensor: A 3-dimensional Tensor with values in the range of uint8 [0, 255].
    """
    # Open the image file
    with Image.open(path) as img:
        # Apply EXIF orientation if needed
        if apply_exif_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation, None)
                    if orientation_value is not None:
                        if orientation_value == 3:
                            img = img.rotate(180, expand=True)
                        elif orientation_value == 6:
                            img = img.rotate(270, expand=True)
                        elif orientation_value == 8:
                            img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # Cases: image don't have getexif
                pass

        # Convert image to the specified mode
        img = img.convert(mode)

        # Convert image to a tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img)

        # Convert to uint8 range [0, 255]
        img_tensor = (img_tensor * 255).to(torch.uint8)

        return img_tensor

