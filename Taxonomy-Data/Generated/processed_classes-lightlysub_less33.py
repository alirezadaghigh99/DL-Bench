import torch
import torch.nn as nn
import torch.nn.functional as F

class SwaVLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_gather_distributed: bool = False,
    ):
        super(SwaVLoss, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_gather_distributed = sinkhorn_gather_distributed

    def subloss(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Apply temperature scaling
        z = z / self.temperature
        
        # Compute log softmax of z
        log_z = F.log_softmax(z, dim=1)
        
        # Compute the cross-entropy loss using KL divergence
        # Note: F.kl_div expects the input to be log probabilities
        # and the target to be probabilities.
        loss = F.kl_div(log_z, q, reduction='batchmean')
        
        return loss