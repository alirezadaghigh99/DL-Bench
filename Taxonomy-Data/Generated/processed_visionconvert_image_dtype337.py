import torch

def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch.Tensor")

    if dtype not in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]:
        raise ValueError("Unsupported dtype. Supported dtypes are: torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64")

    # Handle conversion from float to int types that might cause overflow
    if (image.dtype == torch.float32 and dtype in [torch.int32, torch.int64]) or \
       (image.dtype == torch.float64 and dtype == torch.int64):
        raise RuntimeError(f"Conversion from {image.dtype} to {dtype} might lead to overflow errors.")

    # Define the scale factors for conversion
    def get_scale_factor(src_dtype, dst_dtype):
        if src_dtype.is_floating_point and not dst_dtype.is_floating_point:
            return 255 if dst_dtype == torch.uint8 else 1
        elif not src_dtype.is_floating_point and dst_dtype.is_floating_point:
            return 1 / 255 if src_dtype == torch.uint8 else 1
        return 1

    scale_factor = get_scale_factor(image.dtype, dtype)

    # Convert the image
    if scale_factor != 1:
        image = image * scale_factor

    return image.to(dtype)

