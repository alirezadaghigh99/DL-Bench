import torch
import torch.nn as nn

class QuantizedLayer:
    # Placeholder for the base class
    pass

class QuantizedLinear(QuantizedLayer, nn.Linear):
    def __init__(self, *args, activation_bits=8, requantize_output=True, ema_decay=0.9999, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
        self.register_buffer("input_thresh", torch.zeros(1))
        if self.requantize_output:
            self.register_buffer("output_thresh", torch.zeros(1))

    def quantize(self, tensor, bits, threshold):
        scale = (2 ** bits - 1) / threshold
        return torch.round(tensor * scale).clamp(-2**(bits-1), 2**(bits-1) - 1)

    def dequantize(self, tensor, bits, threshold):
        scale = (2 ** bits - 1) / threshold
        return tensor / scale

    def inference_quantized_forward(self, input):
        assert not self.training, "This function should only be used in inference mode."

        # Quantize the input
        input_quantized = self.quantize(input, self.activation_bits, self.input_thresh.item())

        # Quantize weights and biases
        weight_quantized = self.quantize(self.weight, self.activation_bits, self.input_thresh.item())
        if self.bias is not None:
            bias_quantized = self.quantize(self.bias, self.activation_bits, self.input_thresh.item())
        else:
            bias_quantized = None

        # Perform the linear operation
        output_quantized = torch.nn.functional.linear(input_quantized, weight_quantized, bias_quantized)

        # Dequantize the output
        output_dequantized = self.dequantize(output_quantized, self.accumulation_bits, self.input_thresh.item())

        # Requantize and dequantize the output if required
        if self.requantize_output:
            output_quantized = self.quantize(output_dequantized, self.activation_bits, self.output_thresh.item())
            output_dequantized = self.dequantize(output_quantized, self.activation_bits, self.output_thresh.item())

        return output_dequantized