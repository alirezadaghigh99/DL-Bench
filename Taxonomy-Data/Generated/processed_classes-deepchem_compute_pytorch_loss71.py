import torch
import torch.nn as nn

class Loss:
    # Assuming there are other methods and attributes in the base Loss class
    pass

class L2Loss(Loss):
    def _create_pytorch_loss(self):
        def mse_loss(output, labels):
            # Ensure the shapes of output and labels are consistent
            if output.shape != labels.shape:
                raise ValueError(f"Shape mismatch: output shape {output.shape} and labels shape {labels.shape} must be the same.")
            
            # Compute the mean squared error loss without reduction
            loss = (output - labels) ** 2
            return loss
        
        return mse_loss

