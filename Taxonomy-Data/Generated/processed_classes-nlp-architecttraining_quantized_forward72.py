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

    def fake_quantize(self, tensor, num_bits, threshold):
        # Fake quantization function
        qmin = 0
        qmax = (1 << num_bits) - 1
        scale = threshold / qmax
        quantized = torch.clamp((tensor / scale).round(), qmin, qmax)
        return quantized * scale

    def update_ema(self, current_value, ema_value):
        return self.ema_decay * ema_value + (1 - self.ema_decay) * current_value

    def training_quantized_forward(self, input):
        assert self.training, "training_quantized_forward should only be called during training"

        # Quantize the input
        input_max = input.abs().max()
        if self.input_thresh.item() == 0:
            self.input_thresh.fill_(input_max)
        else:
            self.input_thresh.fill_(self.update_ema(input_max, self.input_thresh.item()))
        
        quantized_input = self.fake_quantize(input, self.activation_bits, self.input_thresh.item())

        # Quantize the weights
        weight_max = self.weight.abs().max()
        quantized_weight = self.fake_quantize(self.weight, self.activation_bits, weight_max)

        # Perform the linear operation with quantized inputs and weights
        output = nn.functional.linear(quantized_input, quantized_weight, self.bias)

        # Optionally requantize the output
        if self.requantize_output:
            output_max = output.abs().max()
            if self.output_thresh.item() == 0:
                self.output_thresh.fill_(output_max)
            else:
                self.output_thresh.fill_(self.update_ema(output_max, self.output_thresh.item()))
            
            output = self.fake_quantize(output, self.activation_bits, self.output_thresh.item())

        return output