import torch
import torch.nn.functional as F
from torch.nn import Module

class SigmoidCrossEntropy(Module):
    def _create_pytorch_loss(self, logits, labels):
        """
        Calculate the sigmoid cross entropy loss between logits and labels.

        Args:
            logits (torch.Tensor): The input logits with shape (batch_size) or (batch_size, tasks).
            labels (torch.Tensor): The target labels with the same shape as logits.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure the labels are of the same type as logits
        labels = labels.type_as(logits)
        
        # Calculate the binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        return loss