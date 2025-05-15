import torch
import torch.nn.functional as F

def _jpeg_encode(image_rgb, jpeg_quality, quantization_table_y, quantization_table_c):
    # Convert RGB to YCbCr
    def rgb_to_ycbcr(image):
        matrix = torch.tensor([[0.299, 0.587, 0.114],
                               [-0.168736, -0.331264, 0.5],
                               [0.5, -0.418688, -0.081312]], dtype=image.dtype, device=image.device)
        shift = torch.tensor([0, 128, 128], dtype=image.dtype, device=image.device)
        image_ycbcr = torch.tensordot(image.permute(0, 2, 3, 1), matrix, dims=1) + shift
        return image_ycbcr.permute(0, 3, 1, 2)

    # Downsample Cb and Cr channels
    def downsample(image):
        return F.avg_pool2d(image, kernel_size=2, stride=2)

    # Perform block splitting and DCT
    def block_dct(image):
        B, C, H, W = image.shape
        image = image.unfold(2, 8, 8).unfold(3, 8, 8)
        image = image.contiguous().view(B, C, -1, 8, 8)
        dct_matrix = torch.tensor([[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
                                   [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                                   [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                                   [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
                                   [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
                                   [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
                                   [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
                                   [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]], dtype=image.dtype, device=image.device)
        dct_matrix_t = dct_matrix.t()
        dct = torch.matmul(dct_matrix, image)
        dct = torch.matmul(dct, dct_matrix_t)
        return dct

    # Quantize the DCT coefficients
    def quantize(dct, quant_table):
        return torch.round(dct / quant_table)

    # Convert RGB to YCbCr
    image_ycbcr = rgb_to_ycbcr(image_rgb)

    # Split into Y, Cb, Cr
    y, cb, cr = image_ycbcr[:, 0, :, :], image_ycbcr[:, 1, :, :], image_ycbcr[:, 2, :, :]

    # Downsample Cb and Cr
    cb_downsampled = downsample(cb.unsqueeze(1)).squeeze(1)
    cr_downsampled = downsample(cr.unsqueeze(1)).squeeze(1)

    # Perform DCT on 8x8 blocks
    y_dct = block_dct(y.unsqueeze(1))
    cb_dct = block_dct(cb_downsampled.unsqueeze(1))
    cr_dct = block_dct(cr_downsampled.unsqueeze(1))

    # Quantize the DCT coefficients
    y_encoded = quantize(y_dct, quantization_table_y)
    cb_encoded = quantize(cb_dct, quantization_table_c)
    cr_encoded = quantize(cr_dct, quantization_table_c)

    return y_encoded, cb_encoded, cr_encoded

