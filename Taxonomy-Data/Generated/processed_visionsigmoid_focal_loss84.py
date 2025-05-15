import torch
import torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='none'):
    """
    Compute the focal loss between `inputs` and the ground truth `targets`.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape representing the predictions.
        targets (Tensor): A float tensor with the same shape as inputs representing binary classification labels.
        alpha (float): A float weighting factor to balance positive vs negative examples.
        gamma (float): A float exponent to balance easy vs hard examples.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.

    Returns:
        Tensor: The computed focal loss.
    """
    # Ensure inputs and targets are of the same shape
    if inputs.shape != targets.shape:
        raise ValueError("Inputs and targets must have the same shape.")

    # Apply the sigmoid function to the inputs
    prob = torch.sigmoid(inputs)
    
    # Compute the binary cross entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    
    # Compute the modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    
    # Compute the alpha factor
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Compute the focal loss
    focal_loss = alpha_factor * modulating_factor * bce_loss
    
    # Apply reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'none':
        return focal_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'none', 'mean', or 'sum'.")

