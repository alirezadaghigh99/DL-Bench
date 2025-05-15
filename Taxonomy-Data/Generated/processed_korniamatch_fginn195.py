import torch
from torch import Tensor
from typing import Tuple, Optional

def match_fginn(
    desc1: Tensor,
    desc2: Tensor,
    lafs1: Tensor,
    lafs2: Tensor,
    th: float = 0.8,
    spatial_th: float = 10.0,
    mutual: bool = False,
    dm: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    # Compute pairwise descriptor distances
    if dm is None:
        dm = torch.cdist(desc1, desc2, p=2)  # Euclidean distance

    # Find the nearest neighbors in desc2 for each vector in desc1
    min_dist, min_idx = torch.min(dm, dim=1)

    # Apply descriptor distance threshold
    valid_matches = min_dist < th

    # Optionally apply spatial threshold
    if spatial_th is not None:
        # Compute spatial distances
        spatial_dist = torch.cdist(lafs1[:, :2, 2], lafs2[:, :2, 2], p=2)
        valid_matches &= spatial_dist[torch.arange(len(min_idx)), min_idx] < spatial_th

    # Filter matches based on the valid matches
    min_dist = min_dist[valid_matches]
    min_idx = min_idx[valid_matches]

    if mutual:
        # Perform mutual nearest neighbor check
        reverse_dm = torch.cdist(desc2, desc1, p=2)
        reverse_min_dist, reverse_min_idx = torch.min(reverse_dm, dim=1)
        mutual_matches = reverse_min_idx[min_idx] == torch.arange(len(min_idx), device=min_idx.device)
        min_dist = min_dist[mutual_matches]
        min_idx = min_idx[mutual_matches]

    return min_dist, min_idx

