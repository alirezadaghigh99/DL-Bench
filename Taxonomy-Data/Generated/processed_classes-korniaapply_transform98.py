import torch
from typing import Dict, Optional

def apply_transform(
    input: torch.Tensor,
    params: Dict[str, torch.Tensor],
    flags: Dict[str, torch.Tensor],
    transform: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Extract the brightness factor from the params dictionary
    brightness_factor = params.get("brightness_factor", torch.tensor(1.0))
    
    # Apply the brightness transformation
    output = input * brightness_factor
    
    # Clip the output if the clip_output flag is set
    if flags.get("clip_output", True):
        output = torch.clamp(output, 0, 1)
    
    return output