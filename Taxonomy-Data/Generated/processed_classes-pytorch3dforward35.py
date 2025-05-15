import torch
import torch.nn as nn
from typing import Optional

class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions: int = 6, omega_0: float = 1.0, logspace: bool = True, append_input: bool = True):
        super(HarmonicEmbedding, self).__init__()
        
        if logspace:
            frequencies = omega_0 * (2.0 ** torch.arange(n_harmonic_functions))
        else:
            frequencies = omega_0 * torch.linspace(1.0, n_harmonic_functions, n_harmonic_functions)
        
        self.register_buffer("_frequencies", frequencies, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor, diag_cov: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute the harmonic embedding
        # x shape: (batch_size, ..., input_dim)
        # frequencies shape: (n_harmonic_functions,)
        
        # Expand dimensions to match x for broadcasting
        frequencies = self._frequencies.unsqueeze(0)  # shape: (1, n_harmonic_functions)
        
        # Compute the sine and cosine components
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, ..., input_dim, 1)
        angles = x_expanded * frequencies  # shape: (batch_size, ..., input_dim, n_harmonic_functions)
        
        sin_components = torch.sin(angles)  # shape: (batch_size, ..., input_dim, n_harmonic_functions)
        cos_components = torch.cos(angles)  # shape: (batch_size, ..., input_dim, n_harmonic_functions)
        
        # Concatenate sine and cosine components
        harmonic_embedding = torch.cat([sin_components, cos_components], dim=-1)  # shape: (batch_size, ..., input_dim, 2 * n_harmonic_functions)
        
        # Optionally append the original input
        if self.append_input:
            harmonic_embedding = torch.cat([x, harmonic_embedding], dim=-1)  # shape: (batch_size, ..., input_dim + 2 * n_harmonic_functions)
        
        # Handle the optional diagonal covariance
        if diag_cov is not None:
            # Assuming diag_cov is of shape (batch_size, ..., input_dim)
            diag_cov_expanded = diag_cov.unsqueeze(-1)  # shape: (batch_size, ..., input_dim, 1)
            harmonic_embedding = harmonic_embedding * diag_cov_expanded  # Element-wise multiplication
        
        return harmonic_embedding