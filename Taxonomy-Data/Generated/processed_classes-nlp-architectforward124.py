import torch

class FakeLinearQuantizationWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits=8):
        # Calculate the quantization range
        qmin = 0
        qmax = 2**bits - 1
        
        # Quantize the input
        input_scaled = input / scale
        input_rounded = torch.round(input_scaled)
        
        # Clamp the values to the quantization range
        input_clamped = torch.clamp(input_rounded, qmin, qmax)
        
        # Dequantize the clamped values
        output = input_clamped * scale
        
        return output