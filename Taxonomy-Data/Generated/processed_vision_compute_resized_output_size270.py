def _compute_resized_output_size(image_size, size=None, max_size=None, allow_size_none=False):
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("image_size must be a list or tuple of two integers (height, width).")
    
    original_height, original_width = image_size

    if size is None:
        if not allow_size_none:
            raise ValueError("size cannot be None unless allow_size_none is True.")
        if max_size is None:
            raise ValueError("max_size must be an integer when size is None.")
        # If size is None and max_size is provided, scale the image to fit within max_size
        if original_height > original_width:
            scale = max_size / float(original_height)
        else:
            scale = max_size / float(original_width)
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        return [new_height, new_width]

    if isinstance(size, int):
        # Resize the smaller edge to 'size' while maintaining aspect ratio
        if original_height < original_width:
            new_height = size
            new_width = int(size * original_width / original_height)
        else:
            new_width = size
            new_height = int(size * original_height / original_width)
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        new_height, new_width = size
    else:
        raise ValueError("size must be an int or a list/tuple of two integers.")

    if max_size is not None:
        if not isinstance(max_size, int):
            raise ValueError("max_size must be an integer.")
        if max(new_height, new_width) > max_size:
            # Scale down to fit within max_size
            scale = max_size / float(max(new_height, new_width))
            new_height = int(new_height * scale)
            new_width = int(new_width * scale)

    return [new_height, new_width]

