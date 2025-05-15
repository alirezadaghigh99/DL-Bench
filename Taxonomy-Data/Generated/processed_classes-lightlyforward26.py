import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(MemoryBankModule):
    def __init__(self, temperature: float = 0.5, memory_bank_size: Union[int, Sequence[int]] = 0, gather_distributed: bool = False):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

    def forward(self, out0: torch.Tensor, out1: torch.Tensor) -> torch.Tensor:
        # Normalize the outputs
        out0 = F.normalize(out0, p=2, dim=1, eps=self.eps)
        out1 = F.normalize(out1, p=2, dim=1, eps=self.eps)

        # Concatenate the outputs
        out = torch.cat([out0, out1], dim=0)

        # Compute the similarity matrix
        sim_matrix = torch.mm(out, out.t().contiguous())
        
        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # Create labels for the positive pairs
        batch_size = out0.size(0)
        labels = torch.arange(batch_size, device=out0.device)
        labels = torch.cat([labels, labels], dim=0)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=out0.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        # Compute the loss
        loss = self.cross_entropy(sim_matrix, labels)

        return loss