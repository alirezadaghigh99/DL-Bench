import torch
from torch import Tensor

def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute the 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: A list of positions to be encoded, of size (M,).

    Returns:
        Position embeddings of size (M, D).

    Raises:
        AssertionError: If *embed_dim* is not even.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Create a tensor of shape (embed_dim // 2,) with values [0, 1, 2, ..., embed_dim // 2 - 1]
    div_term = torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim

    # Compute the sine and cosine terms
    angle_rates = pos.unsqueeze(1) * (10000 ** div_term)
    pos_embedding = torch.zeros((pos.size(0), embed_dim), dtype=torch.float32)
    pos_embedding[:, 0::2] = torch.sin(angle_rates)  # Apply sine to even indices
    pos_embedding[:, 1::2] = torch.cos(angle_rates)  # Apply cosine to odd indices

    return pos_embedding