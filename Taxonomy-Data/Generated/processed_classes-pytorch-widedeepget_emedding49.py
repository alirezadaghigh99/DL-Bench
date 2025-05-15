import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple, Literal

class BayesianTabMlp(BaseBayesianModel):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: Optional[bool] = None,
        cont_embed_dim: Optional[int] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        use_cont_bias: Optional[bool] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "leaky_relu",
        prior_sigma_1: float = 1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
        pred_dim=1,
    ):
        super(BayesianTabMlp, self).__init__()
        
        # Initialize categorical embedding layers
        if cat_embed_input is not None:
            self.cat_embed = nn.ModuleList([
                nn.Embedding(num_categories, embed_dim)
                for _, num_categories, embed_dim in cat_embed_input
            ])
        
        # Initialize continuous normalization and embedding layers
        if continuous_cols is not None:
            if cont_norm_layer == "batchnorm":
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            elif cont_norm_layer == "layernorm":
                self.cont_norm = nn.LayerNorm(len(continuous_cols))
            else:
                self.cont_norm = None
            
            if embed_continuous:
                self.cont_embed = nn.Linear(len(continuous_cols), cont_embed_dim)
            else:
                self.cont_embed = None

    def _get_embeddings(self, X: Tensor) -> Tensor:
        embeddings = []
        
        # Process categorical features
        if hasattr(self, 'cat_embed'):
            cat_embeddings = []
            for i, embed in enumerate(self.cat_embed):
                cat_idx = X[:, i].long()  # Assuming categorical features are at the start
                cat_embeddings.append(embed(cat_idx))
            cat_embeddings = torch.cat(cat_embeddings, dim=1)
            embeddings.append(cat_embeddings)
        
        # Process continuous features
        if hasattr(self, 'cont_norm') or hasattr(self, 'cont_embed'):
            cont_start_idx = len(self.cat_embed) if hasattr(self, 'cat_embed') else 0
            continuous_data = X[:, cont_start_idx:]
            
            if hasattr(self, 'cont_norm') and self.cont_norm is not None:
                continuous_data = self.cont_norm(continuous_data)
            
            if hasattr(self, 'cont_embed') and self.cont_embed is not None:
                continuous_data = self.cont_embed(continuous_data)
            
            embeddings.append(continuous_data)
        
        # Concatenate all embeddings
        if embeddings:
            return torch.cat(embeddings, dim=1)
        else:
            return X  # Return X if no embeddings are created
