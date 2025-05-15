import torch
import torch.nn.functional as F

class SoftmaxCrossEntropy:
    def _compute_pytorch_loss(self, output, labels):
        """
        Computes the softmax cross entropy loss between output logits and labels.

        Parameters:
        - output (torch.Tensor): The output logits with shape (batch_size, classes) or (batch_size, tasks, classes).
        - labels (torch.Tensor): The ground truth labels with the same shape as output.

        Returns:
        - loss (torch.Tensor): The computed loss value.
        """
        # Check if the output is 3D (i.e., has tasks dimension)
        if output.dim() == 3:
            # Reshape output and labels to 2D for cross entropy computation
            batch_size, tasks, classes = output.shape
            output = output.view(batch_size * tasks, classes)
            labels = labels.view(batch_size * tasks)
        else:
            # Ensure labels are in the correct shape for 2D output
            labels = labels.view(-1)

        # Compute the cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss