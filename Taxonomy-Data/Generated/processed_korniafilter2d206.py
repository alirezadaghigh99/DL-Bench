import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr'):
    # Validate input dimensions
    if len(input.shape) != 4:
        raise ValueError("Input tensor must have shape (B, C, H, W)")
    
    if len(kernel.shape) not in [3]:
        raise ValueError("Kernel must have shape (1, kH, kW) or (B, kH, kW)")

    B, C, H, W = input.shape
    kB, kH, kW = kernel.shape

    # Normalize the kernel if required
    if normalized:
        kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)

    # Handle padding
    if padding == 'same':
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    # Apply padding
    if border_type == 'constant':
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    elif border_type == 'reflect':
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    elif border_type == 'replicate':
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    elif border_type == 'circular':
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    else:
        raise ValueError("Invalid border_type. Expected 'constant', 'reflect', 'replicate', or 'circular'.")

    # Flip the kernel for convolution if behaviour is 'conv'
    if behaviour == 'conv':
        kernel = torch.flip(kernel, dims=(-2, -1))

    # Expand kernel to match input channels
    kernel = kernel.expand(B, C, kH, kW)

    # Perform convolution
    output = torch.zeros_like(input)
    for b in range(B):
        for c in range(C):
            output[b, c] = F.conv2d(input_padded[b, c].unsqueeze(0).unsqueeze(0), 
                                    kernel[b, c].unsqueeze(0).unsqueeze(0), 
                                    padding=0).squeeze()

    return output

