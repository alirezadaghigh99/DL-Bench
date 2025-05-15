import torch
import torch.nn as nn
from typing import List

class SwaVLoss(nn.Module):
    def __init__(self):
        super(SwaVLoss, self).__init__()

    def subloss(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Assuming subloss is a cross-entropy loss between z and q
        # z: predictions, q: target codes
        return nn.functional.cross_entropy(z, q)

    def forward(self, high_resolution_outputs: List[torch.Tensor], low_resolution_outputs: List[torch.Tensor], queue_outputs: List[torch.Tensor] = None) -> torch.Tensor:
        num_views = len(high_resolution_outputs)
        total_loss = 0.0
        count = 0

        # Iterate over each view
        for i in range(num_views):
            # High-resolution to low-resolution
            for j in range(num_views):
                if i != j:
                    z_i = high_resolution_outputs[i]
                    q_j = low_resolution_outputs[j]
                    total_loss += self.subloss(z_i, q_j)
                    count += 1

            # Optionally include queue outputs
            if queue_outputs is not None:
                for queue_output in queue_outputs:
                    total_loss += self.subloss(high_resolution_outputs[i], queue_output)
                    count += 1

        # Average the total loss over the number of sublosses computed
        final_loss = total_loss / count if count > 0 else torch.tensor(0.0, device=high_resolution_outputs[0].device)

        return final_loss