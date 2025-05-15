import torch
import torch.nn.functional as F
from torch.nn import Loss

class CategoricalCrossEntropy(Loss):
    def _create_pytorch_loss(self, probabilities, labels):
        """
        Calculate the categorical cross-entropy loss between output probabilities and labels.

        Args:
            probabilities (torch.Tensor): The predicted probabilities with shape (batch_size, classes)
                                          or (batch_size, tasks, classes).
            labels (torch.Tensor): The true labels with the same shape as probabilities.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Check if the input is 3D (batch_size, tasks, classes)
        if probabilities.dim() == 3:
            # Reshape to (batch_size * tasks, classes) for cross-entropy calculation
            batch_size, tasks, classes = probabilities.shape
            probabilities = probabilities.view(batch_size * tasks, classes)
            labels = labels.view(batch_size * tasks, classes)
        
        # Calculate the cross-entropy loss
        # Note: F.cross_entropy expects class indices, not one-hot encoded labels
        # So, we need to convert labels from one-hot to class indices
        labels = torch.argmax(labels, dim=-1)
        
        # Compute the loss
        loss = F.cross_entropy(probabilities, labels)
        
        return loss