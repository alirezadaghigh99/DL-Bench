import torch
import torch.distributed as dist

class VICRegLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed: bool = False,
        eps=0.0001,
    ):
        super(VICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if self.gather_distributed and dist.is_initialized():
            z_a = self.gather_from_all_gpus(z_a)
            z_b = self.gather_from_all_gpus(z_b)

        invariance = invariance_loss(z_a, z_b)
        variance = variance_loss(z_a, self.eps) + variance_loss(z_b, self.eps)
        covariance = covariance_loss(z_a) + covariance_loss(z_b)

        loss = (
            self.lambda_param * invariance +
            self.mu_param * variance +
            self.nu_param * covariance
        )
        return loss

    def gather_from_all_gpus(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gather tensors from all GPUs
        tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        return torch.cat(tensors_gather, dim=0)

def invariance_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)

def variance_loss(x: torch.Tensor, eps: float) -> torch.Tensor:
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(torch.relu(1 - std))

def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    n, d = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (n - 1)
    cov_diag = torch.diagonal(cov)
    off_diag_cov = cov - torch.diag(cov_diag)
    return (off_diag_cov ** 2).sum() / d