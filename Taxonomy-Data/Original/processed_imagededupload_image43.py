def load_image(
    image_file: Union[PurePath, str],
    target_size: Tuple[int, int] = None,
    grayscale: bool = False,
    img_formats: List[str] = IMG_FORMATS,
) -> np.ndarray:
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.

    Args:
        image_file: Path to the image file.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded.
    """
    try:
        img = Image.open(image_file)

        # validate image format
        if img.format not in img_formats:
            logger.warning(f'Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')

            img = preprocess_image(img, target_size=target_size, grayscale=grayscale)

            return img

    except Exception as e:
        logger.warning(f'Invalid image file {image_file}:\n{e}')
        return None